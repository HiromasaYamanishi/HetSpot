import torch
import torch.nn as nn
from torch_geometric.nn import Linear
from torch_geometric.nn.conv import HANConv
#from attention import AttentionModule
#from heterolinear import HeteroLinear
#from get_data import get_data
import torch.nn.functional as F

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

class HAN(torch.nn.Module):
    def __init__(self, data, config, out_channels=1, multi=True,):
        super().__init__()
        self.hidden_channels = config['model']['hidden_channels']
        self.num_layers = config['model']['num_layers']
        self.concat = config['model']['concat']
        self.ReLU = config['model']['ReLU']
        self.config = config
        
        self.multi = multi
        if multi==True:
            self.att = AttentionModule(input_dim=512, split=5)
        
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(HANConv(in_channels=-1, out_channels=self.hidden_channels, metadata=data.metadata(), heads=8))
        
        in_channels_dict = {node_type: self.hidden_channels for node_type in data.x_dict.keys()}
        self.linears = HeteroLinear(in_channels_dict, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict_all = {node_type: [] for node_type in x_dict.keys()}
        if self.multi==True:
            x_dict['spot'] = self.att(x_dict['spot'])

        for l in self.layers:
            x_dict = {k: F.dropout(v, p=self.config['model']['dropout'], training=self.training)
                      for k,v in x_dict.items()}
            x_dict = l(x_dict, edge_index_dict)
            if not self.concat:continue
            for node_type in x_dict.keys():
                x_dict_all[node_type].append(x_dict[node_type])
            if self.ReLU:
                x_dict = {node_type: x.relu() for node_type, x in x_dict.items()}   
        out_dict = self.linears(x_dict)
        if self.concat:
            #x_dict = {node_type: torch.cat(x, dim=1) for node_type, x in x_dict_all.items()}
            x_dict = {node_type: torch.mean(torch.stack(x, dim=1), dim=1) for node_type, x in x_dict_all.items()}
        return x_dict, out_dict

if __name__=='__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data = get_data(category=True, city=True, prefecture=True, multi=True)
    
    model = HAN(data, num_layers=5,hidden_channels=128, out_channels=1,multi=True)
    data.to(device)
    model.to(device)
    print(model)
    x_dict, out_dict= model(data.x_dict, data.edge_index_dict)
    print(x_dict)