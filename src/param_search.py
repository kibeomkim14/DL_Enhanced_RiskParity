
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
from network import GPCopulaRNN
from utils import transform, inv_transform, gaussian_loss, sequence_sampler

# import feature 
prices = pd.read_csv(config.DATA_PATH+'/prices.csv', index_col='Date')

# preprocess weekly data
prices.index = pd.to_datetime(prices.index)
prices = prices.resample('M').last()
lret_m = np.log(prices/prices.shift(1)).fillna(0.0) # weekly return


def train(
        model:nn.Module, 
        data:torch.Tensor, 
        optimizer:torch.optim.Optimizer, 
        num_samples:int,
        num_epochs:int
    ) -> torch.Tensor:

    model.init_weight()

    # transform 
    X_tr, cdfs = transform(data)
    sampler = sequence_sampler(tr_idx=data.size(0), context_len=12, 
                                prediction_len=1, num_samples=num_samples)
        
    for epoch in range(num_epochs):
        # randomly sample sequence index for training
        losses = []
        losses_valid = torch.Tensor([0.0])
        for tr_idx, te_idx in sampler: # for each sequence sample
            optimizer.zero_grad()

            # run the model
            mu_t, cov_t, batch_idx = model(data[tr_idx])
            x = X_tr[tr_idx][:,batch_idx]

            # gaussian log-likelihood loss
            loss = gaussian_loss(x, mu_t, cov_t)
            loss.backward()
            optimizer.step()
            
            # append loss of each batch
            losses.append(loss.detach())

            with torch.no_grad():
                # run the model
                mu_t, _, _ = model(data[tr_idx], pred=True)
                z_hat = inv_transform(mu_t[-1], cdfs)
                
                # prediction step using last time-step prediction
                mu_pred, _, _ = model(z_hat.unsqueeze(0), pred=True)
                Z_tr_hat = inv_transform(mu_pred.detach(), cdfs)
                
            # calculate validation loss
            losses_valid = losses_valid + (Z_tr_hat[0]- data[te_idx]).pow(2).sum()

        losses_valid = losses_valid.item()/num_samples
        if epoch+1 % 25 == 0:
            print(f'Epoch {epoch+1}') 
            print(f'Gaussian LL Loss : {np.round(np.mean(losses),2)}')
            print(f'Validation MSE   : {losses_valid} \n')
    return losses_valid

def test(model: nn.Module,
         data: torch.Tensor, 
         cdfs:dict
        ) -> torch.Tensor:
    loss = torch.Tensor([0.0])
    count = 0
    for i in range(6,data.size(0)):
        z, z_targ = data[i-6:i], data[i:i+1]    
        count += 1
        with torch.no_grad():
            # run the model
            mu_t, _, _ = model(z, pred=True)
            z_hat = inv_transform(mu_t[-1], cdfs)
        
            # use sampled z hat for next step prediction
            mu_pred, _, _ = model(z_hat.unsqueeze(0), pred=True)
            Z_tr_hat = inv_transform(mu_pred.detach(), cdfs)
        
        loss = loss + (Z_tr_hat - z_targ).pow(2).sum()
    loss = loss/count
    print(f'Prediction MSE : {loss}')
    return loss
    
def objective(trial: optuna.Trial) -> float:
    # set up the parameter using optuna
    LEARNING_RATE = trial.suggest_float("learning rate",1e-6,5e-3,log=True)
    WEIGHT_DECAY  = trial.suggest_float("weight decay" ,1e-6,3e-2,log=True)
    
    # define the GP-Copula model and initialize the weight
    # set up the loss function and optimizer
    model = GPCopulaRNN(input_size=1, hidden_size=4, num_layers=2, rank_size=5, 
                        batch_size=3, num_asset=7, dropout=0.1, features=config.e_dict)
    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # convert the data to torch tensor (70% train)
    split_idx = int(lret_m.shape[0] * 0.7)
    Z_tr = torch.Tensor(lret_m.iloc[:split_idx].values)

    # train the model and store network parameters in Trial user attribute
    losses_valid = train(model, Z_tr, optimizer, config.NUM_SAMPLES, config.NUM_EPOCHS)
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
