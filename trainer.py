from __future__ import print_function, division
import matplotlib.pyplot as plt
import time
import copy
import math
from typing import OrderedDict

import os
import sys
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn import ReLU
import pandas as pd
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch.nn.functional import relu
from torch_geometric.nn import GATConv,HGTConv, GCNConv, HANConv, SAGEConv, HeteroConv, GATv2Conv
from torch_geometric.nn import Linear, to_hetero, Sequential
from torch_geometric.loader import NeighborLoader, HGTLoader
from preprocessing import Path
from utils import save_plot, save_cor, EarlyStopping
from torch_scatter.scatter import scatter
from torch.nn import LayerNorm
from dataloader import get_data
from torchmetrics import MeanAbsolutePercentageError
import wandb

class Trainer:
    def __init__(self, config,):
        self.config = config
        self.device = config['device']
        self.df = pd.read_csv('./data/spot/experience_light.csv')
        self.spot_names = self.df['spot_name'].values
        self.max_cor = 0
        self.W = torch.nn.parameter.Parameter(torch.rand(128, 128))
        torch.nn.init.normal_(self.W, 0.1)
        self.W=self.W.to(self.device)
        print(self.W.device)
        print('loaded neg samples')
        self.city_loss_weight = self.config['city_pop_weight']
        self.mse = torch.nn.MSELoss()
        self.mape = MeanAbsolutePercentageError().to(config['device'])
        self.mae = torch.nn.L1Loss()
        self.best_val_mse = 1e9
        
        self.checkpoint_path = './checkpoints/' + self.config['model']['model_type'] + '_' \
                                + str(self.config['model']['num_layers']) + '_' \
                                + str(self.config['model']['hidden_channels']) + '.pth'
                                
        self.early_stopping = EarlyStopping(patience=50, path=self.checkpoint_path)
        if config['use_wandb']:
            wandb.init('popularity', config=config)

    def train(self, model, optimizer, data, epoch):
        model.train()
        optimizer.zero_grad()
        x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
        mask = data['spot'].train_mask
        total_loss = 0
        if isinstance(x_dict, dict):
            spot_mask = data['spot'].train_mask
            loss_spot = F.mse_loss(out_dict['spot'][spot_mask].flatten(), data['spot'].y[spot_mask])
            loss = loss_spot
            loss = loss.float()
            total_loss+=loss

        if self.config['use_wandb']:
            wandb.log({'loss':loss})

        loss.backward()
        optimizer.step()
        return model,float(total_loss)


    @torch.no_grad()
    def test(self, model,data):
        model.eval()
        x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
        # losses = []
        losses = defaultdict(dict)
        for split in ['train_mask', 'valid_mask', 'test_mask']:
            mask = data['spot'][split]
            if isinstance(x_dict, dict):
                spot_mask = data['spot'][split]
                mse = self.mse(out_dict['spot'][spot_mask].flatten(), data['spot'].y[spot_mask])
                rmse = torch.sqrt(mse)
                mape = self.mape(out_dict['spot'][spot_mask].flatten(), data['spot'].y[spot_mask])
                mae = self.mae(out_dict['spot'][spot_mask].flatten(), data['spot'].y[spot_mask])
                
                losses[split]['mse'] = mse
                losses[split]['rmse'] = rmse
                losses[split]['mape'] = mape
                losses[split]['mae'] = mae
                loss_city=0
                if self.config['data']['city']:
                    city_mask = data['city'][split]
                    loss_city = F.mse_loss(out_dict['city'][city_mask].flatten(), data['city'].y[city_mask])/mask.sum()
                
            else:
                loss = F.mse_loss(out_dict[mask].flatten(), data['spot'].y[mask])/mask.sum()
        return losses

    def train_epoch(self, model, data, epoch_num):
        optimizer = optim.Adam(model.parameters(), lr=self.config['trainer']['lr'])
        for epoch in range(epoch_num):
            model, loss = self.train(model, optimizer, data, epoch)
            losses= self.test(model,data)
            mse_train, mse_valid, mse_test = losses['train_mask']['mse'], losses['valid_mask']['mse'],losses['test_mask']['mse']
            rmse_train, rmse_valid, rmse_test = losses['train_mask']['rmse'], losses['valid_mask']['rmse'],losses['test_mask']['rmse']
            mape_train, mape_valid, mape_test = losses['train_mask']['mape'], losses['valid_mask']['mape'],losses['test_mask']['mape']          
            mae_train, mae_valid, mae_test = losses['train_mask']['mae'], losses['valid_mask']['mae'],losses['test_mask']['mae']
            print(f'Epoch: {epoch+1:03d}, MSE train: {mse_train:.4f}', f'Val: {mse_valid:.4f}', f'Test: {mse_test:.4f}')
            print(f'Epoch: {epoch+1:03d}, RMSE train: {rmse_train:.4f}', f'Val: {rmse_valid:.4f}', f'Test: {rmse_test:.4f}')
            print(f'Epoch: {epoch+1:03d}, MAPE train: {mape_train:.4f}', f'Val: {mape_valid:.4f}', f'Test: {mape_test:.4f}')
            print(f'Epoch: {epoch+1:03d}, MAE train: {mae_train:.4f}', f'Val: {mae_valid:.4f}', f'Test: {mae_test:.4f}')
            self.early_stopping(mse_valid, model)
            if mse_valid<self.best_val_mse:
                self.best_val_mse = mse_valid
                torch.save(model.state_dict(), self.checkpoint_path)
            
            if epoch%10==9:
                cor = self.calc_cor(model, data)
                print('cor is ', cor)
                if cor>self.max_cor:
                    self.max_cor = cor
                    self.best_epoch = epoch
                    self.calc_cor_save(model, data)
                    
            if self.early_stopping.early_stop:
                print('Early Stop')
                break
                    
        # final result
        model.load_state_dict(torch.load(self.checkpoint_path))
        losses = self.test(model, data)
        test_mse, test_rmse, test_mape, test_mae = \
            losses['test_mask']['mse'], losses['test_mask']['rmse'], losses['test_mask']['mape'], losses['test_mask']['mae']
        cor = self.calc_cor(model, data)
        self.best_mse, self.best_rmse, self.best_mape, self.best_mae =\
            test_mse.item(), test_rmse.item(), test_mape.item(), test_mae.item()
        self.best_cor = cor
        print(f'cor: {cor:.4f}')

    @torch.no_grad()
    def calc_cor(self, model, data):
        model.eval()
        x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
        for split in ['test_mask']:
            mask = data['spot'][split]
            gt = data['spot'].y[mask].cpu().numpy()
            if isinstance(out_dict, dict):
                pred = out_dict['spot'][mask].reshape(-1).cpu().numpy()
            else:
                pred = out_dict[mask].reshape(-1).cpu().numpy()
            cor = np.corrcoef(gt, pred)[0][1]
        return cor

    @torch.no_grad()
    def calc_cor_save(self, model, data):
        print('saving cor')
        model.eval()
        gt_all_spot, pred_all_spot=[], []
        gt_all_city, pred_all_city = [], []
        x_dict, out_dict = model(data.x_dict, data.edge_index_dict)
        path = Path()

        for split in ['test_mask']:
            spot_mask = data['spot'][split]
            gt_spot = data['spot'].y[spot_mask]

            if isinstance(out_dict, dict):
                pred_spot = out_dict['spot'][spot_mask].flatten()
            else:
                mask = data['spot'][split]
                pred = out[mask]
            gt_all_spot.append(gt_spot.cpu().detach().numpy().copy())
            pred_all_spot.append(pred_spot.cpu().detach().numpy().copy())

        gt_all_spot = np.concatenate(gt_all_spot)
        pred_all_spot = np.concatenate(pred_all_spot).reshape(-1)
        self.save_cor(gt_all_spot, pred_all_spot,'gt: log(review_count)','pred: log(review_count)','cor')
        return np.corrcoef(gt_all_spot, pred_all_spot)[0][1], 0 #np.corrcoef(gt_all_city, pred_all_city)[0][1]

    def save_cor(self, x, y, x_name, y_name,save_name,*args):
        plt.rcParams["font.size"] = 18
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, aspect='equal')
        ax.set_xlim(0,4)
        ax.set_ylim(0,4)
        ax.set_xlabel(x_name, fontsize=18)
        ax.set_ylabel(y_name, fontsize=18)
        cor=np.corrcoef(x, y)[0][1]
        ax.set_title(f'cor: {round(cor,5)}', fontsize=18)
        ax.scatter(x, y)
        fig.subplots_adjust(bottom = 0.15)
        plt.savefig(f'{save_name}.png')
        if len(args)>0:
            f = open('out_spot.txt', 'w')
            for i, arg in enumerate(args[0]):
                if abs(x[i]-y[i])>1.5:
                    ax.annotate(arg, (x[i],y[i]), fontname='Noto Serif CJK JP')
                    f.write(arg)
                    f.write(f' gt:{x[i]}')
                    f.write(f' pred:{y[i]}')
                    f.write('\n')
            f.close()
            plt.savefig('cor_with_name.png')
            

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = get_data(category=True, city=True, prefecture=True, multi=True)

    data.to(device)
    print(data)
    
    model = MyHetero(data.x_dict, data.edge_index_dict, num_layers=4, hidden_channels=128, out_channels=1, out_dim=512,multi=True)
    model.to(device)
    
    trainer = Trainer(device)
    trainer.train_epoch(model, data, epoch_num=1500)