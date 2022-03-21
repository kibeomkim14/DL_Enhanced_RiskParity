import torch
import torch.nn as nn
from typing import Tuple
from torch.distributions.multivariate_normal import MultivariateNormal


class GPCopulaNet(nn.Module):
    def __init__(
        self, 
        input_dim :int, 
        hidden_dim:int, 
        embed_dim :int,
        num_layers:int,
        num_assets:int,
        seq_length:int,
        pred_length:int,
        batch_size:int,
        rank_size:int,
        dropout:float=0.1,
        batch_first:bool=False,
    ):
        torch.set_default_dtype(torch.float64)
        super(GPCopulaNet,self).__init__()
        self.input_dim  = input_dim  
        self.hidden_dim = hidden_dim 
        self.embed_dim  = embed_dim 
        self.num_layers = num_layers
        self.num_assets = num_assets
        self.seq_length = seq_length 
        self.pred_length= pred_length
        self.batch_size = batch_size
        self.rank_size  = rank_size
        self.batch_first = batch_first
        self.hidden = {}

        # local lstm
        self.embed = nn.Embedding(self.num_assets, self.embed_dim)
        self.lstm  = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                             num_layers=num_layers, batch_first=batch_first, dropout=dropout)

        # define linear weight for calculating parameters of gaussian process
        self.layer_m = nn.Linear(self.hidden_dim + self.embed_dim, 1) # mean 
        self.layer_v = nn.Linear(self.hidden_dim + self.embed_dim, self.rank_size) # diagonal point               
        self.layer_d = nn.Sequential(
                                        nn.Linear(self.hidden_dim + self.embed_dim, 1),
                                        nn.Softplus(beta=1)
                                    ) # volatility

    def init_weight(self):
        for i in range(self.num_assets):
            self.hidden['asset_'+str(i)] = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),\
                                            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, inputs:torch.Tensor, indices:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        INPUTS
            inputs  : torch.Tensor.Size(sequence_length x num_sampled_assets)
            indices : torch.Tensor.Size(num_sampled_assets)
        OUTPUTS
            mu_t  : torch.Tensor.Size(sequence_length x num_sampled_assets x 1)
            cov_t : torch.Tensor.Size(sequence_length x num_sampled_assets x num_sampled_assets)
        """
        mus, vs, ds = [], [], []
        for idx in indices: # for each asset
            output, (h, c)  = self.lstm(inputs.unsqueeze(1)[:,:,idx:idx+1], self.hidden['asset_'+str(idx.item())])
            self.hidden['asset_'+str(idx.item())] = (h.detach(), c.detach()) # store hidden state

            # concatenate h_it with embeddings e_i
            embedding = self.embed(torch.ones(inputs.size(0)).type(torch.long) * idx)
            y = torch.concat([output, embedding.unsqueeze(1)], axis=2)

            # calculate mean and variances of asset i
            mus.append(self.layer_m(y).view(inputs.size(0),1))
            ds.append(self.layer_d(y).view(inputs.size(0),1))
            vs.append(self.layer_v(y).view(inputs.size(0),self.rank_size))

        mu_t, d_t, v_t = torch.stack(mus, axis=1), torch.stack(ds, axis=1), torch.stack(vs, axis=1)
        cov_t = torch.diag_embed(d_t.squeeze(2)) + (v_t @ v_t.permute(0,2,1)) # D + V @ V.T
        return mu_t, cov_t

    def predict(self, z:torch.Tensor, num_samples:int=10):
        self.eval()
        x_samples  = []
        hidden_original = self.hidden.copy()
        
        with torch.no_grad():
            for i in range(num_samples):
                input = z
                trajectory = []
                for t in range(self.pred_length):
                    mu, cov = self.forward(input, torch.arange(self.num_assets))
                    distrib = MultivariateNormal(mu.view(-1), cov)
                    sample = distrib.sample((1,)).squeeze(0)
                    trajectory.append(sample.view(-1))
                    input = sample
                x_samples.append(torch.stack(trajectory, axis=0))
                    
        # revert hidden states to the point before prediction range.
        self.hidden = hidden_original
        return torch.stack(x_samples,axis=0)