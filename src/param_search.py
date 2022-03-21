
import torch 
import torch.nn as nn
import pandas as pd
import optuna
import argparse
import logging

from config import *
from torch.optim import Adam
from network import GPCopulaNet
from torch.utils.data import DataLoader
from loader import TrainDataset, EvalDataset
from utils import train_test_split, transform, inv_transform_3D, loss_GLL


def train(
        model:nn.Module,
        data:pd.DataFrame, 
        optimizer:torch.optim.Optimizer, 
        num_epochs:int,
        num_samples:int
    ) -> torch.Tensor:

    logging.getLogger('GPCopula.Train')

    valid_split = int(data.shape[0] * 0.8)
    df_tr, df_vl = data.iloc[:valid_split], data.iloc[valid_split-SEQ_LENGTH:]
    dataset_tr, dataset_te = TrainDataset(df_tr, context_len=SEQ_LENGTH), EvalDataset(df_vl, context_len=SEQ_LENGTH)
    _, cdf_tr = transform(torch.Tensor(df_tr.values))
    loader = DataLoader(dataset_tr, batch_size=1)

    # initialize weight and put the model on training mode.
    model.init_weight()
    model.train()

    for epoch in range(num_epochs):
        loss = torch.Tensor([0.0])
        for n_step, (train_batch, _) in enumerate(loader):
            train_batch, indice = train_batch.squeeze(0), torch.randperm(7)[:3]
            mu, cov = model(train_batch, indice)
            x, _ = transform(train_batch, cdf_tr)
            loss = loss + loss_GLL(x[:,indice], mu, cov)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print results
        if epoch % 5 == 0 and epoch != 0: 
            print(f'Epoch {epoch} Training GLL Loss (sum): {loss.item()/(n_step+1)}')
        if epoch+1 == 11 or epoch+1 == num_epochs:
            print('Validating with validation data...\n')
            valid_loss = evaluate(model, dataset_te, cdf_tr, num_samples)
    return valid_loss

def evaluate(
        model:nn.Module,
        dataset:torch.utils.data.Dataset, 
        cdf:dict,
        num_samples:int
    ) -> torch.Tensor:
    
    logging.getLogger('GPCopula.Eval')
    loss_fn = nn.MSELoss()

    model.eval()
    pred, truth = [], []
    loader = DataLoader(dataset, batch_size=1)

    for _, (test_batch, label) in enumerate(loader):
        test_batch, indice = test_batch.squeeze(0), torch.arange(7)
        _, _ = model(test_batch, indice)

        x_hat = model.predict(test_batch[-2:-1], num_samples)
        z_hat = inv_transform_3D(x_hat, cdf)
        z_mean, _ = z_hat.mean(axis=0), z_hat.std(axis=0)
        pred.append(z_mean)
        truth.append(label.squeeze(0))

    pred  = torch.stack(pred, axis=0)
    truth = torch.stack(truth, axis=0)
    loss = loss_fn(pred, truth)
    print(f'Validation MSE: {loss.item()}')
    return loss.item()
    
def objective(trial: optuna.Trial) -> float:
    # set up the parameter using optuna
    LEARNING_RATE = trial.suggest_float("learning rate",5e-5,1e-1,log=True)
    WEIGHT_DECAY  = trial.suggest_float("weight decay" ,5e-5,1e-1,log=True)
    
    # set up the data
    # load and preprocess the data
    feature = pd.read_csv(DATA_PATH+'log_return.csv', index_col='Date')
    df_tr, _ = train_test_split(feature)
    
    # set up network and optimizer
    model = GPCopulaNet(input_dim=1, hidden_dim=15, embed_dim=6, num_layers=2,
                    num_assets=7, seq_length=24, pred_length=4, batch_size=1, rank_size=2)
    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # train the model and store network parameters in Trial user attribute
    losses_valid = train(model, df_tr, optimizer, NUM_SAMPLES, NUM_EPOCHS)
    trial.set_user_attr('net_params', {'net_params':model.state_dict()})
    return losses_valid

if __name__ == "__main__":
    # Initialize ArgParser
    parser = argparse.ArgumentParser()
    
    # add argument
    parser.add_argument("--n_trials" ,type=int)
    args = parser.parse_args()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)
    print('Hyperparameter search complete.')
    
    # store parameters
    parameters = study.best_trial.user_attrs['net_params']
    hyp_params = study.best_trial.params
    parameters['hyp_params'] = hyp_params
    
    print('saving parameters...')
    torch.save(parameters, 'models/GPCopulaNet.pt')
