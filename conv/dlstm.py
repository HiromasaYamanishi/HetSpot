import torch
import torch.nn as nn
from torch_geometric.nn import Linear
from collections import defaultdict
from torch_scatter.scatter import scatter
#from attention import AttentionModule
#from heterolinear import HeteroLinear
import sys
#from get_data import get_data
import yaml
import torch.nn.functional as F
from torch_geometric.nn.aggr.lstm import LSTMAggregation
from torch_geometric.nn.aggr import MaxAggregation

class AttentionModule(torch.nn.Module):
    def __init__(self, input_dim, num_heads=4, split=1,):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.split = split
        self.out_dim = input_dim
        self.per_dim = input_dim//num_heads

        self.W = torch.nn.ModuleList([Linear(input_dim, self.per_dim, False, weight_initializer='glorot') for _ in range(num_heads)])
        self.q = torch.nn.ParameterList([])
        for _ in range(num_heads):
            q_ =torch.nn.Parameter(torch.zeros(size=(self.per_dim, 1)))
            nn.init.xavier_uniform_(q_.data, gain=1.414)
            self.q.append(q_)
        
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        out = []
        x = x.resize(x.size()[0],self.split, self.input_dim)
        for i in range(self.num_heads):
            W = self.W[i]
            q = self.q[i]
            x_ = W(x)
            att = self.LeakyReLU(torch.matmul(x_, q))
            att = torch.nn.functional.softmax(att, dim=1)
            att = torch.broadcast_to(att, x_.size())
            x_= (x_*att).sum(dim=1)
            out.append(x_)
        return torch.cat(out, dim=1)

class HeteroLinear(torch.nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super().__init__()
        self.linears = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.linears[node_type] = Linear(in_channels, out_channels, weight_initializer='glorot')

    def forward(self, x_dict):
        x_dict_out = {}
        for node_type, x in x_dict.items():
            x = self.linears[node_type](x)
            x_dict_out[node_type] = x
        return x_dict_out

class HeteroLSTMConv(torch.nn.Module):
    def __init__(self, in_channels_dict, edge_index_dict, out_channels, ReLU, config):
        super().__init__()

        self.linear = nn.ModuleDict({})
        self.div = defaultdict(int)
        for k in edge_index_dict.keys():
            self.linear['__'.join(k) + '__source'] = Linear(in_channels_dict[k[0]], out_channels, False, weight_initializer='glorot')
            self.linear['__'.join(k) + '__target'] = Linear(in_channels_dict[k[-1]], out_channels, False, weight_initializer='glorot')
            self.div[k[-1]]+=1

        self.lstm = nn.ModuleDict({})
        for k in edge_index_dict.keys():
            self.lstm['__'.join(k)] = nn.LSTMCell(out_channels, out_channels)

        self.ReLU = ReLU
        self.config = config
        self.aggr = LSTMAggregation(out_channels, out_channels)
        self.aggr = MaxAggregation()
        #self.aggr_dict = nn.ModuleDict({})
        #for k in edge_index_dict.keys():
        #    self.aggr_dict['__'.join(k)] = LSTMAggregation(out_channels, out_channels)
        
    def forward(self, x_dict, edge_index_dict):
        x_dict_out = {}
        target_tmp={}
        aggregated_tmp = {}
        aggregate_meta={}
        for k,v in edge_index_dict.items():
            source, target = k[0], k[-1]
            source_x = self.linear['__'.join(k) + '__source'](x_dict[source])
            target_x = self.linear['__'.join(k) + '__target'](x_dict[target])
            source_index = v[0].reshape(-1)
            target_index = v[1].reshape(-1)
            out = torch.zeros_like(target_x).to(target_x.device)
            source_x = source_x[source_index]

            #target_x = target_x + scatter(source_x, target_index, out=out, dim=0, reduce='mean')
            source_x = F.dropout(source_x, p=self.config['model']['dropout'], training = self.training)
            #aggr = self.aggr_dict['__'.join(k)]
            aggr = self.aggr
            aggregated = self.aggr(source_x, target_index)

            target_tmp[k]=target_x
            aggregated_tmp[k]=aggregated
            #print(aggregated.shape)
            if aggregate_meta.get(target)!=None:
                aggregate_meta[target]+=aggregated
            else:
                aggregate_meta[target]=aggregated
        aggregate_meta = {k: v/self.div[k] for k,v in aggregate_meta.items()}
        if self.config['model']['meta']:
            x_dict_out_tmp = {k: self.lstm['__'.join(k)](target_tmp[k], (aggregated_tmp[k], aggregate_meta[k[-1]]))[0] for k in edge_index_dict.keys()}
        else:
            x_dict_out_tmp = {k: self.lstm['__'.join(k)](target_tmp[k], (aggregated_tmp[k], aggregated_tmp[k]))[0] for k in edge_index_dict.keys()}
        for k,v in x_dict_out_tmp.items():
            if x_dict_out.get(k[-1])!=None:
                x_dict_out[k[-1]]+=v
            else:
                x_dict_out[k[-1]]=v
        #x_dict_out = {k: self.l2_norm(v) for k,v in x_dict_out.items()}    

        x_dict_out = {k: v/self.div[k] for k,v in x_dict_out.items()}   
        if self.ReLU:
            x_dict_out = {k: v.relu() for k,v in x_dict_out.items()} 
        return x_dict_out


class HeteroDLSTM(torch.nn.Module):
    def __init__(self, data, config, out_channels=1,multi=True):
        super().__init__()
        self.hidden_channels = config['model']['hidden_channels']
        self.num_layers = config['model']['num_layers']
        self.concat = config['model']['concat']
        self.ReLU = config['model']['ReLU']

        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        self.layers = torch.nn.ModuleList()
        self.multi = multi
        self.first_in_channels_dict = {node_type: x.size(1) for node_type, x in x_dict.items()}
        self.mid_in_channels_dict = {node_type: self.hidden_channels for node_type in x_dict.keys()}
        if multi==True:
            self.att = AttentionModule(input_dim=512, split=5)
            self.first_in_channels_dict['spot'] = 512
       
        self.layers.append(HeteroLSTMConv(self.first_in_channels_dict, edge_index_dict, self.hidden_channels, self.ReLU, config))

        for i in range(self.num_layers-1):
            self.layers.append(HeteroLSTMConv(self.mid_in_channels_dict, edge_index_dict, self.hidden_channels, self.ReLU, config))
        self.linears = HeteroLinear(self.mid_in_channels_dict, out_channels)
        self.multi = multi

    def forward(self, x_dict, edge_index_dict):
        x_dict_all = {node_type: [] for node_type in x_dict.keys()}
        if self.multi==True:
            x_dict['spot'] = self.att(x_dict['spot'])

        for l in self.layers:
            x_dict = l(x_dict, edge_index_dict)
            if not self.concat:continue
            for node_type in x_dict.keys():
                x_dict_all[node_type].append(x_dict[node_type])
        
        out_dict = self.linears(x_dict)
        if self.concat==True:
            #x_dict = {node_type: torch.cat(x, dim=1) for node_type, x in x_dict_all.items()}
            x_dict = {node_type: torch.mean(torch.stack(x, dim=1), dim=1) for node_type, x in x_dict_all.items()}
        return x_dict, out_dict

if __name__=='__main__':
    with open('../config.yaml') as f:
        config = yaml.safe_load(f)
    device = 'cuda:1'
    config['k'] = 20
    config['device'] = device
    config['explain_num'] = 10
    config['epoch_num'] = 2500
    config['model']['model_type'] = 'ggnn'
    config['model']['num_layers'] = 3
    config['model']['hidden_channels'] = 128
    config['model']['concat'] = True
    config['model']['ReLU'] = True
    config['model']['dropout'] = 0.1
    config['trainer']['explain_span'] = 50
    config['trainer']['lr'] = 0.0003
    config['trainer']['loss_city_weight'] = 0
    config['trainer']['loss_category_weight'] = 0
    config['trainer']['loss_word_weight'] = 0.2
    config['trainer']['loss_pref_weight'] = 0
    config['trainer']['city_pop_weight']=0
    config['trainer']['spot_pop_weight']=0
    config['data']['word'] = True
    config['data']['city'] = True
    config['data']['category'] = True
    config['data']['prefecture'] = False
    config['model']['meta'] = False
    data = get_data(category=True, city=True, prefecture=False, multi=True)
    
    model = HeteroDLSTM(data, config)
    data.to(device)
    model.to(device)
    all_param = 0
    for p in model.parameters():
        all_param+=torch.norm(p)
        print(torch.norm(p))
    print(all_param)
    exit()
    print(torch.sum(model.parameters()))
    print(model)
    x_dict, out_dict= model(data.x_dict, data.edge_index_dict)
    optim = torch.optim.Adam(model.parameters())
    y = torch.rand(42852).to(device)
    loss = torch.nn.functional.mse_loss(y, out_dict['spot'])
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(x_dict)