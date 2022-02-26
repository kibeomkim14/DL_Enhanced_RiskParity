import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import PARA_PATH
from network import MLP
from torch.utils.data import Dataset

def train_test_split(data:pd.DataFrame, valid_ratio:float=0.2, train_ratio:float=0.6):
    # we will use 60% of the data as train+val set. this data will be used to tune DL model.
    n = data.shape[0]
    tr_idx  = int(n * train_ratio)
    val_idx = int(tr_idx * (1-valid_ratio))

    # split the dataset according to tr index
    tr_data = data.iloc[:tr_idx]
    te_data = data.iloc[tr_idx:]

    # Now we create a target for training a DL model. Target will be a future 130-days return of each ETF products.
    # To make a target we look 130 days ahead of the training set. This will be done by shifting the view by 130 days.
    target_list = ['BNDX_ret130','BND_ret130','VGK_ret130','VNQI_ret130',
                    'VNQ_ret130','VTI_ret130','VWOB_ret130','VWO_ret130']
    target = data.iloc[130:tr_idx+130][target_list]

    # add prefix on columns and reset the
    target = target.add_prefix('targ_')
    target.index = tr_data.index

    assert tr_data.shape[0] == target.shape[0], 'feature shape and target shape is not same.'
    mu, std = tr_data.mean(axis=0), tr_data.std(axis=0)
    tr_data = tr_data.sub(mu).div(std) # normalize the data along rows

    trn_features, trn_label = tr_data.iloc[:val_idx-130], target.iloc[:val_idx-130] # cut last few days of data to prevent data leakage.
    val_features, val_label = tr_data.iloc[val_idx:], target.iloc[val_idx:]

    return trn_features, trn_label, val_features, val_label, te_data

def train_test_split2(data:pd.DataFrame, test:bool=True):
    # preprocess the data for training
    y = data.iloc[130:][['BNDX_ret130','BND_ret130','VGK_ret130','VNQI_ret130','VNQ_ret130','VTI_ret130','VWOB_ret130','VWO_ret130']]
    y.index = data.index[:-130]
    X = data.loc[:'2021-08-18'] 

    if test:
        # split train and test data
        tr_X, tr_y = X.loc[:'2016-02-22'], y.loc[:'2016-02-22']
        te_X, te_y = X.loc['2016-02-23':], y.loc['2016-02-23':]

        # normalize the inputs
        mu, std = tr_X.mean(axis=0), tr_X.std(axis=0)
        tr_X, te_X = tr_X.sub(mu).div(std), te_X.sub(mu).div(std) 
        return tr_X, tr_y, te_X, te_y
    else:
        tr_X, tr_y = X, y
        mu, std = tr_X.mean(axis=0), tr_X.std(axis=0)
        tr_X = tr_X.sub(mu).div(std)
        return tr_X, tr_y

def generate_portfolio_inputs(feature:pd.DataFrame, target:pd.DataFrame) -> pd.DataFrame:
    data  = torch.Tensor(feature.values.copy())
    label = torch.Tensor(target.values.copy())

    # load trained parameters
    model = MLP()
    net_params = torch.load(PARA_PATH)['net_state_dict']
    model.load_state_dict(net_params)
    
    # MC Dropout 
    predictions = []
    for _ in range(1000):
        predictions.append(model(data).detach())
    predictions = torch.stack(predictions)

    # calculate bayesian uncertainty estimates
    exp_returns = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)
    error = (label - exp_returns).abs()

    # concatenate and append
    meta_features = torch.concat([exp_returns, uncertainty, error], axis=1)
    meta_features = pd.DataFrame(meta_features.numpy(), index=feature.index)

    # rename columns 
    meta_features.columns = [name.split('_')[0] + '_exp_return' for name in target.columns.values]  + \
                            [name.split('_')[0] + '_uncertainty' for name in target.columns.values] + \
                            [name.split('_')[0] + '_mae' for name in target.columns.values] 

    # calculate exponential average of mae features
    meta_features.iloc[:,-8:] = meta_features.iloc[:,-8:].ewm(10).mean()
    return meta_features

def sharpe_ratio(weight:torch.Tensor, lret:torch.Tensor) -> torch.Tensor:
    portfolio_return = (weight * lret).sum(axis=1) - 1
    mean_return = portfolio_return.mean() * torch.tensor(252)
    volatility  = portfolio_return.std() * torch.sqrt(torch.tensor(252))
    return -mean_return/volatility

class InvDataset(Dataset):
    def __init__(self, features:torch.Tensor, returns:torch.Tensor):
        self.features = features
        self.returns  = returns

    def __len__(self) -> int:
        return self.features.size(0)
    
    def __getitem__(self, idx):
        X_selected = self.features[idx]
        y_selected = self.returns[idx]
        return X_selected, y_selected
