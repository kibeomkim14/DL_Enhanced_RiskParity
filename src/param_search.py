
import torch 
import torch.nn as nn
import optuna
import pandas as pd
import numpy as np
import argparse
import config

from optuna import trial
from network import GPCopulaRNN
from folds import PurgedKFold
from utils import train_test_split, transform, inv_transform, \
                    gaussian_loss, train_idx_sampler

# import feature 
prices = pd.read_csv(config.DATA_PATH+'/prices.csv', index_col='Date')

# preprocess weekly data
prices.index = pd.to_datetime(prices.index)
prices = prices.resample('M').last()
lret_m = np.log(prices/prices.shift(1)).fillna(0.0) # weekly return


def objective(trial: optuna.Trial) -> float:
    # set up the parameter using optuna
    LEARNING_RATE = trial.suggest_float("learning rate",1e-6,5e-3,log=True)
    WEIGHT_DECAY  = trial.suggest_float("weight decay" ,1e-6,3e-2,log=True)
    
    # define the GP-Copula model and initialize the weight
    model = GPCopulaRNN(input_size=1, hidden_size=4, num_layers=2, rank_size=4, 
                        batch_size=3, num_asset=7, dropout=0.05, features=config.e_dict)
    model.init_weight()

    # set up the loss function and optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    num_epochs  = 500
    num_samples = 20

    # convert the data to torch tensor
    split_idx = int(lret_m.shape[0] * 0.7)
    Z_tr, Z_te = torch.Tensor(lret_m.iloc[:split_idx].values), torch.Tensor(lret_m.iloc[split_idx:].values)

    # transform 
    X_tr, cdfs = transform(Z_tr)
    X_te, _    = transform(Z_te, cdfs)


    for epoch in range(num_epochs):
        # randomly sample sequence index for training
        sampler = train_idx_sampler(tr_idx=Z_tr.size(0), context_len=12, prediction_len=1, num_samples=num_samples)
        losses  = []
        losses_valid = torch.Tensor([0.0])

        for tr_idx, te_idx in sampler: # for each sequence sample
            optimizer.zero_grad()

            # run the model
            mu_t, cov_t, batch_idx = model(Z_tr[tr_idx])
            x = X_tr[tr_idx][:,batch_idx]

            # gaussian log-likelihood loss
            loss = gaussian_loss(x, mu_t, cov_t)
            loss.backward()
            optimizer.step()
            
            # append loss of each batch
            losses.append(loss.detach())

            # prediction step
            mu_pred, _, _ = model(Z_tr[te_idx], pred=True)
            Z_tr_hat = inv_transform(mu_pred.detach(), cdfs)
            
            # calculate validation loss
            losses_valid = losses_valid + (Z_tr_hat[0]- Z_tr[te_idx]).pow(2).sum()

        losses_valid = losses_valid[0]/num_samples
        if epoch % 50 == 0:
            print(f'Epoch {epoch+1}') 
            print(f'Gaussian LL Loss : {np.round(np.mean(losses),2)}')
            print(f'Validation MSE   : {losses_valid} \n')
    return losses_valid


if __name__ == "__main__":
    # Initialize ArgParser
    parser = argparse.ArgumentParser()
    
    # add argument
    parser.add_argument("--n_trials",type=int)
    args = parser.parse_args()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)
    params = study.best_trial.params

    print('Hyperparameter search complete. ')
    print(params)
    #torch.save({'net_state_dict':model.state_dict(), 'optimizer_dict':optimizer.state_dict()}, 'models/forecaster_params.json')
