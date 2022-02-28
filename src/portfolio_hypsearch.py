from network import MLP, PortfolioLayer
from config import DATA_PATH, PARAM_PATH
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import Dataset, DataLoader
from utils import train_test_split2, generate_portfolio_inputs, sharpe_ratio, InvDataset
from optuna import trial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import argparse

# import data and trained network parameters
pf_feature = pd.read_csv(DATA_PATH+'dropout_feature.csv', index_col='Date')
prices     = pd.read_csv(DATA_PATH+'prices.csv', index_col='Date')

# calculate log return
prices = prices.loc[pf_feature.index]
lret = np.log(prices/prices.shift()).fillna(0).add(1)


def objective(trial: optuna.Trial) -> float:
    # set up the parameter using optuna
    LEARNING_RATE = trial.suggest_float("learning rate",1e-6,5e-3,log=True)
    WEIGHT_DECAY  = trial.suggest_float("weight decay" ,1e-6,3e-2,log=True)
    NUM_EPOCHS = 50
    
    # define dataset
    X = pf_feature.loc[:'2015-09-25'].values
    y = lret.loc[:'2015-09-25'].values
    assert X.shape[0] == y.shape[0], 'dataset not equal'

    # define a portfolio layer
    # in this training scheme, we use Walk-forward validation to train portfolio model.
    # Expanding window was used in this case.
    cv  = TimeSeriesSplit(n_splits=10,test_size=150)
    portfolio = PortfolioLayer(asset_num=8, error_adjusted=True)
    optimizer = optim.Adam(params=portfolio.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    val_losses = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_tr, y_tr = torch.Tensor(X[train_idx]), torch.Tensor(y[train_idx])
        X_vl, y_vl = torch.Tensor(X[val_idx]), torch.Tensor(y[val_idx])

        dataset = InvDataset(X_tr,y_tr)
        loader = DataLoader(dataset, batch_size=60, shuffle=False)
        
        for epoch in range(NUM_EPOCHS):
            # training
            for x_batch, y_batch in loader:    
                optimizer.zero_grad()
                weights = portfolio(x_batch)
                loss = sharpe_ratio(weights, y_batch)
                loss.backward()
                optimizer.step()
            
            # at the end of the training, evaluate 
            if epoch + 1 == NUM_EPOCHS:
                with torch.set_grad_enabled(False):
                    weights = portfolio(X_vl)
                    val_loss = -sharpe_ratio(weights, y_vl)
                    val_losses.append(val_loss.detach().item())
                    print(f'Fold: {fold+1}, Validation Sharpe: {val_loss.detach()}')

    return np.mean(val_losses)


if __name__ == "__main__":
    # Initialize ArgParser
    parser = argparse.ArgumentParser()
    
    # add argument
    parser.add_argument("--n_trials",type=int)
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)
    default_params = study.best_trial.params

    print('Hyperparameter search complete. ')
    print('Now train the model given these hyperparameters...')
    
    NUM_EPOCHS = 50

    # define dataset
    X = pf_feature.loc[:'2015-09-25'].values
    y = lret.loc[:'2015-09-25'].values
    
    assert X.shape[0] == y.shape[0], 'dataset not equal'

    # define a portfolio layer
    cv  = TimeSeriesSplit(n_splits=10,test_size=200)
    portfolio = PortfolioLayer(asset_num=8, error_adjusted=True)
    optimizer = optim.Adam(params=portfolio.parameters(), lr=default_params['learning rate'], weight_decay=default_params['weight decay'])

    val_losses = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_tr, y_tr = torch.Tensor(X[train_idx]), torch.Tensor(y[train_idx])
        X_vl, y_vl = torch.Tensor(X[val_idx]), torch.Tensor(y[val_idx])

        # define dataloader for batch gradient descent
        dataset = InvDataset(X_tr,y_tr)
        loader = DataLoader(dataset, batch_size=60, shuffle=False)
        
        for epoch in range(NUM_EPOCHS):
            for x_batch, y_batch in loader: # training
                optimizer.zero_grad()
                weights = portfolio(x_batch)
                loss = sharpe_ratio(weights, y_batch)
                loss.backward()
                optimizer.step()
            
            # at the end of the training, evaluate 
            if epoch + 1 == NUM_EPOCHS:
                with torch.set_grad_enabled(False):
                    weights = portfolio(X_vl)
                    val_loss = -sharpe_ratio(weights, y_vl)
                    val_losses.append(val_loss.detach().item())
                    print(f'Fold: {fold+1}, Validation Sharpe: {val_loss.detach()}')

        print('saving the network parameters...')
        torch.save({'network_parameter':portfolio.state_dict(), 'optimizer_dict':optimizer.state_dict()}, 'models/portfolio_params.json')
