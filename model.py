import torch
from conv.ggnn import HeteroGGNN
from conv.hgt import HGT
from conv.han import HAN
from conv.sage import HeteroSAGE
from conv.sageattn import HeteroSAGEAttention
from conv.lstm import HeteroLSTM
from conv.gcn import HeteroGCN
from conv.rgat import HeteroRGAT
from conv.rgcn import HeteroRGCN
from conv.gat import HeteroGAT
from conv.rggnn import HeteroRGGNN
from dataloader import get_data

class Model(torch.nn.Module):
    def __init__(self, data, config):
        super().__init__()
        self.model_dict = {'sage': HeteroSAGE,
                            'sageattn': HeteroSAGEAttention,
                            'han': HAN,
                            'hgt': HGT,
                            'ggnn': HeteroGGNN,
                            'lstm': HeteroLSTM,
                            'gcn': HeteroGCN,
                            'rgat': HeteroRGAT,
                            'rgcn': HeteroRGCN,
                            'gat': HeteroGAT,
                            'rggnn': HeteroRGGNN
                            }

        self.model = self.model_dict[config['model']['model_type']](data, config)

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = get_data(category=True, city=True, prefecture=True, multi=True)
    from collections import defaultdict
    config = defaultdict(dict)
    config['model']['model_type'] = 'hgt'
    config['model']['num_layers'] = 2
    config['model']['hidden_channels'] = 128
    model = Model(data, config)
    data.to(device)
    model.to(device)
    print(model)
    x_dict, out_dict= model(data.x_dict, data.edge_index_dict)
    print(x_dict)