
import torch 
import torch.nn as nn
import optuna
import pandas as pd
import numpy as np
import argparse
import config
import json 

from optuna import trial
from typing import Tuple
from torch.optim import Adam
from network import GP_RNN, GPCopulaRNN
from utils import load_data, train_test_split, transform, inv_transform, gaussian_loss, sequence_sampler


def train(
        model:nn.Module,
        data:Tuple[torch.Tensor, torch.Tensor], 
        optimizer:torch.optim.Optimizer, 
        num_samples:int,
        num_epochs:int,
        is_copula:bool
    ) -> torch.Tensor:

    # initialize weight and put the model on training mode.
    model.init_weight()
    model.train()

    for epoch in range(num_epochs):
        # randomly sample sequence index for training
        losses_tr = torch.Tensor([0.0])
        sampler   = sequence_sampler(tr_idx=data[0].size(0), context_len=18, 
                                prediction_len=1, num_samples=num_samples)

        optimizer.zero_grad()
        for tr_idx, te_idx in sampler: # for each sequence sample
            mu_t, cov_t, batch_idx = model(data[0][tr_idx])
            
            # gaussian log-likelihood loss
            if is_copula:
                x = inv_transform(data[0][tr_idx][:,batch_idx], model.distribution) 
            else:
                x = data[0][tr_idx][:,batch_idx]
            loss = gaussian_loss(x, mu_t, cov_t)
            
            # append loss of each batch
            z_pred, _ = model.predict(data[0][tr_idx], 1)
            # calculate validation loss
            losses_tr = losses_tr + (z_pred[0].view(-1) - data[0][te_idx].view(-1)).pow(2).sum()

        loss.backward()
        optimizer.step()

        losses_tr = losses_tr.item()/num_samples
        if epoch+1 == num_epochs: 
            print(f'Training MSE: {losses_tr} \n')
        
    loss_valid = test(model, data[1], 18)
    return loss_valid

def test(model: nn.Module,
         data: torch.Tensor,
         context_len:int
        ) -> torch.Tensor:
    """
    모델과, 데이터와 context length 를 이용해 validation data 로 loss 를 구한다.
    """
    model.eval()
    loss = torch.Tensor([0.0])
    count = 0
    for i in range(context_len,data.size(0)):
        z, z_targ = data[i-context_len:i], data[i:i+1]    
        count += 1
        with torch.no_grad():
            # run the model
            z_pred, _ = model.predict(z)

        loss = loss + (z_pred.view(-1) - z_targ.view(-1)).pow(2).sum()
    loss = loss/count
    return loss.item()
    
def objective(trial: optuna.Trial, is_copula:bool) -> float:
    # set up the parameter using optuna
    LEARNING_RATE = trial.suggest_float("learning rate",5e-5,1e-1,log=True)
    WEIGHT_DECAY  = trial.suggest_float("weight decay" ,5e-5,1e-1,log=True)
    
    # set up the data
    prices_m = load_data('M')
    Z, _ = train_test_split(prices_m)
    Z_train, Z_valid = Z[:int(Z.size(0) * 0.8)], Z[int(Z.size(0) * 0.8)-18:]
    
    # set up the model
    if is_copula:
        _, cdf = transform(Z_train)
        model = GPCopulaRNN(input_size=1, hidden_size=4, num_layers=2, rank_size=3, 
                        batch_size=3, num_asset=7, dropout=0.1, cdfs=cdf)
    else:
        model = GP_RNN(input_size=1, hidden_size=4, num_layers=2, rank_size=3, 
                        batch_size=3, num_asset=7, dropout=0.1)

    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # train the model and store network parameters in Trial user attribute
    losses_valid = train(model, (Z_train, Z_valid), optimizer, config.NUM_SAMPLES, config.NUM_EPOCHS, is_copula)
    trial.set_user_attr('net_params', {'net_params':model.state_dict()})
    return losses_valid

def str2bool(v):
    """
    출처: https://eehoeskrap.tistory.com/521
    argparse 통해 boolean 값을 구하기 위해 쓰는 부분이다.
    """
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    # Initialize ArgParser
    parser = argparse.ArgumentParser()
    
    # add argument
    parser.add_argument("--n_trials" ,type=int)
    parser.add_argument("--is_copula", help='enable copula process', default=False, type=str2bool)

    args = parser.parse_args()

    print(args.is_copula)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args.is_copula), n_trials=args.n_trials)
    
    print('Hyperparameter search complete. ')
    
    # store parameters
    parameters = study.best_trial.user_attrs['net_params']
    hyp_params = study.best_trial.params
    parameters['hyp_params'] = hyp_params
    
    print('saving parameters...')
    torch.save(parameters, 'models/GP_RNN.pt')
