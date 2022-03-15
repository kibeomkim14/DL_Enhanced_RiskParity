import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple


class MultiVariateGP(nn.Module):
    def __init__(self, input_size:int, rank_size:int):
        super(MultiVariateGP,self).__init__()
        self.input_size = input_size
        self.rank_size  = rank_size

        # define linear weight for calculating parameters of gaussian process
        # these weights are SHARED across all time series.
        self.layer_m = nn.Linear(input_size, 1) # mean 
        self.layer_d = nn.Linear(input_size, rank_size) # diagonal point
        self.layer_v = nn.Sequential(
                                        nn.Linear(input_size, 1),
                                        nn.Softplus(beta=1)
                                    ) # volatility

    def forward(self, h_t:torch.Tensor):
        m_t = self.layer_m(h_t)
        d_t = self.layer_d(h_t)
        v_t = self.layer_v(h_t)
        return m_t, d_t, v_t


class GPCopulaRNN(nn.Module):
    def __init__(
        self, 
        input_size:int, 
        hidden_size:int, 
        num_layers:int, 
        batch_size:int,
        num_asset:int,
        rank_size:int,
        dropout:float=0.1,
        batch_first:bool=False,
        features:Optional[dict]=None 
    ):
        super(GPCopulaRNN,self).__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.batch_size   = batch_size
        self.num_asset    = num_asset
        self.num_layers   = num_layers
        self.batch_first  = batch_first
        self.feature_size = len(features[0]) if features is not None else 0
        self.features = features
        self.hidden   = None
        
        # local lstm
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=batch_first, dropout=dropout)
        
        # Multivariate Gaussian Process
        self.gp = MultiVariateGP(input_size=hidden_size + self.feature_size, rank_size=rank_size)

    def init_weight(self) -> None:
        h0, c0 = torch.zeros(self.num_layers, self.num_asset, self.hidden_size),\
                    torch.zeros(self.num_layers, self.num_asset, self.hidden_size)
        self.hidden = (h0, c0)

    def feature_selector(self, indices:torch.Tensor, time_steps:int) -> torch.Tensor:
        feature = []
        for idx in indices:
            feature.append(self.features[idx.item()].repeat(time_steps,1))
        feature = torch.stack(feature, axis=1)
        return feature

    def forward(self, z_t:torch.Tensor, pred:bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(z_t.size()) == 2, 'input tensor dimension is not equal to 2.'
        z_t = z_t.unsqueeze(2)

        # select the batch then pass it through the unrolled LSTM
        # timestep x num_batch x 1
        output, (hn, cn) = self.lstm(z_t, self.hidden)
        self.hidden = (hn.detach(), cn.detach())
        
        # we only use the subset of batch for training
        # else return all indices
        if pred:
            batch_indices = torch.arange(self.num_asset)
        else:
            batch_indices = torch.randperm(self.num_asset)[:self.batch_size]
        
        # find feature vector e_i for each asset i then concatenate it with LSTM output.
        e = self.feature_selector(batch_indices, z_t.size(0))
        y_t = torch.concat([output[:,batch_indices,:], e], axis=2)

        # get GP parameters
        # calculate parameters of multivariate gaussian process for each time t
        mu_t, v_t, d_t = self.gp(y_t)
        cov_t = torch.diag_embed(d_t.squeeze(2)) + (v_t @ v_t.permute(0,2,1)) # D + V @ V.T
        return mu_t, cov_t, batch_indices
