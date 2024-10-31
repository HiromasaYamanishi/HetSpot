from dataloader import get_data
from model import Model
from trainer import Trainer 
import torch
import os
import csv
import yaml
import argparse

class ExpRunner:
    def __init__(self, config):
        self.config = config
        self.data = get_data(config)
        self.model =  Model(self.data, config).model
        self.trainer = Trainer(config)
        self.device = config['device']
        self.data.to(self.device)
        self.model.to(self.device)
        print(config)

    def run_experiment(self):
        self.trainer.train_epoch(self.model, self.data, epoch_num=self.config['epoch_num'])

        with open('./result/result.csv', 'a') as f:
            writer = csv.writer(f)
        
            writer.writerow([round(self.trainer.max_cor, 5),
                            self.trainer.best_epoch,
                            round(self.trainer.best_cor, 5),
                            round(self.trainer.best_mse, 5),
                            round(self.trainer.best_rmse, 5),
                            round(self.trainer.best_mape, 5),
                            round(self.trainer.best_mae, 5),
                            self.config['model']['model_type'],
                            self.config['model']['num_layers'],
                            self.config['trainer']['lr'],
                            self.config['model']['hidden_channels'],
                            self.config['model']['concat'],
                            self.config['trainer']['city_pop_weight'],
                            self.config['data']['word'],
                            self.config['data']['category'],
                            self.config['data']['city'],
                            self.config['data']['pref'],
                            self.config['model']['dropout']])

    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--word', action='store_true')
    parser.add_argument('--city', action='store_true')
    parser.add_argument('--category', action='store_true')
    parser.add_argument('--pref', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--city_pop_weight', type=float, default=1e-4)
    parser.add_argument('--epoch_num', type=int, default=500)
    parser.add_argument('--model_type', default='ggnn')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_wandb', action='store_true')
    
    args = parser.parse_args()
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        
    config['model']['num_layers'] = args.num_layers
    config['model']['hidden_channels'] = args.hidden_channels
    config['model']['model_type'] = args.model_type
    config['trainer']['lr'] = args.lr
    config['data']['word'] = args.word
    config['data']['city'] = args.city
    config['data']['category'] = args.category
    config['data']['pref'] = args.pref
    config['model']['ReLU'] = True
    config['model']['concat'] = False
    config['model']['dropout'] = args.dropout
    config['model']['meta'] = False
    config['city_pop_weight'] = args.city_pop_weight
    config['epoch_num'] = args.epoch_num
    config['device'] = f'cuda:{args.gpu}'
    config['use_wandb'] = args.use_wandb
    config['log'] = False
    config['k'] = 20
    print(config)
    
    exp = ExpRunner(config)
    exp.run_experiment()
