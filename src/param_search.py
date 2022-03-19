
import torch 
import torch.nn as nn
import optuna
import pandas as pd
import numpy as np
import argparse
import config
import json 

from optuna import trial
from torch.optim import Adam
from network import GP_RNN, GPCopulaRNN
from utils import load_data, train_test_split, transform, inv_transform, gaussian_loss, sequence_sampler


def train(
        model:nn.Module, 
        data:torch.Tensor, 
        optimizer:torch.optim.Optimizer, 
        num_samples:int,
        num_epochs:int
    ) -> torch.Tensor:

    model.init_weight()
    model.train()

    Z_tr, _ = train_test_split(data)
    Z_train, Z_valid = Z_tr[:int(Z_tr.size(0) * 0.8)], Z_tr[int(Z_tr.size(0) * 0.8)-18:]
    sampler = sequence_sampler(tr_idx=Z_train.size(0), context_len=18, 
                                prediction_len=1, num_samples=num_samples)
    
    for epoch in range(num_epochs):
        # randomly sample sequence index for training
        losses_tr = torch.Tensor([0.0])
        for tr_idx, te_idx in sampler: # for each sequence sample
            optimizer.zero_grad()

            # run the model
            mu_t, cov_t, batch_idx = model(Z_train[tr_idx])

            # gaussian log-likelihood loss
            loss = gaussian_loss(Z_train[tr_idx][:,batch_idx], mu_t, cov_t)
            loss.backward()
            optimizer.step()
            
            # append loss of each batch
            z_pred, _ = model.predict(Z_train[tr_idx], 1)
            
            # calculate validation loss
            losses_tr = losses_tr + (z_pred[0].view(-1) - Z_train[te_idx].view(-1)).pow(2).sum()

        losses_tr = losses_tr.item()/num_samples
        if epoch+1 == num_epochs: 
            print(f'Training MSE: {losses_tr} \n')
        
    loss_valid = test(model, Z_valid, 18)
    return loss_valid

def test(model: nn.Module,
         data: torch.Tensor,
         context_len:int
        ) -> torch.Tensor:
    
    model.eval()
    loss = torch.Tensor([0.0])
    count = 0
    for i in range(context_len,data.size(0)):
        z, z_targ = data[i-context_len:i], data[i:i+1]    
        count += 1
        with torch.no_grad():
            # run the model
            mu_t, _, _ = model(z, pred=True)
            
            # use sampled z hat for next step prediction
            mu_pred, _, _ = model(mu_t[-1].unsqueeze(0), pred=True)
        loss = loss + (mu_pred.view(-1) - z_targ.view(-1)).pow(2).sum()
    loss = loss/count
    return loss.item()
    
def objective(trial: optuna.Trial) -> float:
    # set up the parameter using optuna
    LEARNING_RATE = trial.suggest_float("learning rate",5e-5,1e-1,log=True)
    WEIGHT_DECAY  = trial.suggest_float("weight decay" ,5e-5,1e-1,log=True)
    
    prices_m = load_data('M')

    # define the GP-Copula model and initialize the weight
    # set up the loss function and optimizer
    model = GP_RNN(input_size=1, hidden_size=4, num_layers=2, rank_size=3, 
                        batch_size=3, num_asset=7, dropout=0.1)
    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # train the model and store network parameters in Trial user attribute
    losses_valid = train(model, prices_m, optimizer, config.NUM_SAMPLES, config.NUM_EPOCHS)
    trial.set_user_attr('net_params', {'net_params':model.state_dict()})
    return losses_valid


if __name__ == "__main__":
    # Initialize ArgParser
    parser = argparse.ArgumentParser()
    
    # add argument
    parser.add_argument("--n_trials",type=int)
    args = parser.parse_args()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)
    
    print('Hyperparameter search complete. ')
    
    # store parameters
    parameters = study.best_trial.user_attrs['net_params']
    hyp_params = study.best_trial.params
    parameters['hyp_params'] = hyp_params
    
    print('saving parameters...')
    torch.save(parameters, 'models/GaussCopula.pt')
