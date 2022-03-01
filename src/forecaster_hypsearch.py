
import torch 
import torch.nn as nn
import optuna
import pandas as pd
import numpy as np
import argparse
import config

from network import MLP
from utils import train_test_split
from folds import PurgedKFold
from optuna import trial

# import feature 
feature = pd.read_csv(config.DATA_PATH+'features.csv', index_col='Date')
feature.index = pd.to_datetime(feature.index)
tr_x, tr_y, val_x, val_y = train_test_split(feature)


def objective(trial: optuna.Trial) -> float:
    # set up the parameter using optuna
    LEARNING_RATE = trial.suggest_float("learning rate",1e-6,5e-3,log=True)
    WEIGHT_DECAY  = trial.suggest_float("weight decay" ,1e-6,3e-2,log=True)
    EPSILON = trial.suggest_float("epsilon" ,1e-8, 0.2)

    # call the network for training
    model = MLP()
    model.weight_init()

    # set up the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=EPSILON)

    t1 = pd.Series(feature[20:tr_x.shape[0]+20].index, index=tr_x.index) # look ahead of 130 days
    cv = PurgedKFold(n_splits=5, t1=t1, pctEmbargo=0.05) 

    train_losses = []
    for fold_idx, (tr, val) in enumerate(cv.split(tr_x)): # 5-fold Purged Cross Validation
        # assign training set and validation set of CV scheme
        x_train, y_train = torch.Tensor(tr_x.values[tr]), torch.Tensor(tr_y.values[tr])
        x_valid, y_valid = torch.Tensor(tr_x.values[val]), torch.Tensor(tr_y.values[val])
        
        # for each epoch train and update the network
        for epoch in range(config.NUM_EPOCHS):
            optimizer.zero_grad()
            
            # get prediction and its loss with gradients
            output = model(x_train)
            loss = loss_fn(output, y_train)
            loss.backward()

            # backpropagation
            optimizer.step()
            
        # validate model
        output = model(x_valid)
        valid_loss = loss_fn(output, y_valid)  
        train_losses.append(valid_loss.detach().item())  

    # after training and CV
    x_test, y_test = torch.Tensor(val_x.values), torch.Tensor(val_y.values)
    output = model(x_test).detach()
    test_loss = loss_fn(output, y_test)
    print(f'Average Training loss : {np.mean(train_losses)}, Test Loss: {test_loss}')
    return test_loss


if __name__ == "__main__":
    # Initialize ArgParser
    parser = argparse.ArgumentParser()
    
    # add argument
    parser.add_argument("--n_trials",type=int)
    args = parser.parse_args()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)
    default_params = study.best_trial.params

    print('Hyperparameter search complete. ')
    print('Now train the model given these hyperparameters...')
    # call the network for training
    model = MLP()
    model.weight_init()

    # set up the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=default_params['learning rate'], weight_decay=default_params['weight decay'], eps=default_params['epsilon'])

    t1 = pd.Series(feature[20:tr_x.shape[0]+20].index, index=tr_x.index) # look ahead of 130 days
    cv = PurgedKFold(n_splits=5, t1=t1, pctEmbargo=0.05) 

    train_losses = []
    for fold_idx, (tr, val) in enumerate(cv.split(tr_x)): # 5-fold Purged Cross Validation
        model.train()
        # assign training set and validation set of CV scheme
        x_train, y_train = torch.Tensor(tr_x.values[tr]), torch.Tensor(tr_y.values[tr])
        x_valid, y_valid = torch.Tensor(tr_x.values[val]), torch.Tensor(tr_y.values[val])
        
        # for each epoch train and update the network
        for epoch in range(50):
            optimizer.zero_grad()
            
            # get prediction and its loss with gradients
            output = model(x_train)
            loss = loss_fn(output, y_train)
            loss.backward()

            # backpropagation
            optimizer.step()
            
        # validate model
        model.eval()
        output = model(x_valid)
        valid_loss = loss_fn(output, y_valid)  
        train_losses.append(valid_loss.detach().item())  
    
    # after training and CV
    x_test, y_test = torch.Tensor(val_x.values), torch.Tensor(val_y.values)
    model.eval()
    output = model(x_test).detach()
    test_loss = loss_fn(output, y_test)
    print(f'Average Training loss : {np.mean(train_losses)}, Test Loss: {test_loss}')

    print('saving the network parameters...')
    torch.save({'net_state_dict':model.state_dict(), 'optimizer_dict':optimizer.state_dict()}, 'models/forecaster_params.json')
