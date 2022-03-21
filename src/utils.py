import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from scipy.stats import norm
from typing import Optional, Tuple
from distribution import ECDF_
from torch.distributions.multivariate_normal import MultivariateNormal


def train_test_split(df:pd.DataFrame, split_year:int=2016) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
    학습 데이터와 시험 데이터를 나눠주는 함수입니다. split_ratio 를 이용해서 원하는 비율로 나눌 수 있습니다.
    splits a dataset into train and test sets given a split ratio.

    INPUT
        df: pd.DataFrame
            monthly price of assets
        split_ratio: float
            specifies the ratio of train set over the whole.
    RETURNS
        df_tr, df_te: Tuple[pd.DataFrame,pd.DataFrame]
            returns train & test data
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    # convert the data to torch tensor
    df_tr, df_te = df.loc[:str(split_year)], df.loc[str(split_year+1):]
    return df_tr, df_te


def transform(Z:torch.Tensor, cdfs:Optional[dict]=None) -> Tuple[torch.Tensor,dict]:
    """
    transforms data distribution to standard normal distribution. 
    
    Transform takes 2 steps:
    1. estimate empirical distribution of the data per asset. Then convert it to uniform, u [0,1] distribution
        note that we use step function based empirical CDF which is different to the one specified in the original paper.
        ** values are truncated to prevent standard normal variable goes either -inf or inf.

    2. use inverse CDF of standard normal to convert u to x, where x follows standard normal distribution.
    
      (1)  (2)
    z -> u -> x 

    INPUTS
        df: pd.DataFrame
            input data sequence, that is multivariate
        context_len: Optional[int]
            specifies context length for the training set. rest of the data will be used for prediction.
    
    RETURNS
        X: pd.DataFrame
            returns transformed dataset.
    """
    
    X = []
    m = Z.size(0)
    emp_distributions = {}
    
    # lower and upper bound for emp. CDF
    delta_m = (4 * np.sqrt(np.log(m) * np.pi) * m ** 0.25) ** -1 
    data = torch.clone(Z).numpy()

    for i in range(data.shape[1]):
        # estimate empirical CDF
        # only use m datapoint to estimate CDF.
        Z_i = data[:,i]
        if cdfs is not None:
            emp_cdf = cdfs['CDF_'+str(i)]
        else:
            emp_cdf = ECDF_(Z_i.reshape(-1))
        
        # for each datapoint, transform
        U_i = emp_cdf(Z_i)
        
        # truncate extreme values
        U_i = np.where(U_i < delta_m, delta_m, U_i) 
        U_i = np.where(U_i > 1-delta_m, 1-delta_m, U_i)
        
        # get standard normal values
        X_i = norm.ppf(q=U_i, loc=0, scale=1) # inverse CDF of standard normal
        X.append(torch.Tensor(X_i))
        emp_distributions['CDF_'+str(i)] = emp_cdf
        
    # make a dataframe
    X = torch.stack(X, axis=0).T
    return X, emp_distributions


def inv_transform(X:torch.Tensor, cdfs:dict) -> torch.Tensor:
    """
    inverse transforms standard normal distribution to the original distribution given a set of
    empirical cdfs.
    
    Transform takes 2 steps:
    1. use standard normal to convert x to u, where u is in [0,1].
    2. the input cdfs contains empirical cdfs with its inverse. Using inverse of these cdfs, we transform u to z.
    
      (1)  (2)
    x -> u -> z

    INPUTS
        X: torch.Tensor
            input data sequence, that is multivariate
        cdfs: dict
            a dictionary of empirical cdfs indexed by asset number.
    
    RETURNS
        Z: torch.Tensor
            returns original dataset.
    """ 
    Z = []

    for i in range(X.size(1)):
        X_i = X[:,i].numpy()
        
        # for each datapoint, transform to u by applying standard normal CDF
        U_i = norm.cdf(X_i)
        
        # get empirical distribution of each asset.
        emp_cdf = cdfs['CDF_'+str(i)]

        # transform the uniform data to Z by inverse empirical CDF
        Z_i = emp_cdf.inverse(U_i) 
        Z.append(torch.Tensor(Z_i))

    Z = torch.stack(Z, axis=0).T
    return Z


def inv_transform_3D(X:torch.Tensor, cdfs:dict) -> torch.Tensor:
    """
    inverse transform the 3D Tensor given empirical CDF class. For more details, see the function above.
    """
    assert len(X.size()) == 3, \
        'Tensor dimension is not equal to 3. Use inv_transform function instead.'

    Z = []
    for i in range(X.size(0)):
        Z.append(inv_transform(X[i], cdfs))
    return torch.stack(Z, axis=0)


def sequence_sampler(tr_idx:int, context_len:int, prediction_len:int, num_samples:int) -> list:
    """
    Given a train index (a point), with context length and prediction length, this function samples a sequence of length 
    context_len + prediction_len for 'num_samples' times.
    
    INPUTS
        tr_idx:int
            an end point of training dataset. It is an integer
        context_len:int
            specifies the length of training interval. 
        prediction_len:int
            specifies the length of prediction interval. 
        num_samples:
            specifies the number of samples to be sampled
    
    RETURNS
        sample_indices: list
            a list of sampled indices each in the form of (training indices, prediction indices)
    """
    sample_indices = []
    last_idx = None
    for idx in np.random.randint(0,tr_idx - context_len, size=num_samples):
        if last_idx == idx:
            continue
        sample_indices.append((torch.LongTensor(idx + np.arange(context_len)), \
                            torch.LongTensor(idx + np.arange(prediction_len)+ context_len)))
        last_idx = idx
    return sample_indices


def loss_GLL(x:torch.Tensor, mu_t:torch.Tensor, cov_t:torch.Tensor, type='sum') -> torch.Tensor:
    """
    calculates the log-loglikelihood value of multivariate gaussian process given true value
    x and loc, scale parameters.
    """
    assert x.size(0) == mu_t.size(0), \
        'sequence length is not equal (input and mu_t)'
    assert mu_t.size(0) == cov_t.size(0), \
        'sequence length is not equal (mu_t and cov_t)'    
    assert len(cov_t.size()) == 3, \
        'dimension of covariance matrix is not equal to 3.'

    if len(mu_t.size()) == 3:
        mu_t = mu_t.squeeze(2)

    # set a multivariate distribution indexed by time t
    # loc: batch_size x dim, cov: batch_size x dim x dim
    # input x: batch_size x dim
    distribution_t = MultivariateNormal(mu_t, cov_t)
    loglikelihood = distribution_t.log_prob(x)
    
    # calculate MAPE
    mape = ((mu_t - x).abs()/x.abs())
    mape = torch.where(mape > 100, torch.ones(mape.size()) * 100, mape)
    #+ mape.detach().sum() 
    return -loglikelihood.sum() if type=='sum' else -loglikelihood.mean() # negative log-likelihood + MAPE


def sharpe_ratio(weight:torch.Tensor, lret:torch.Tensor) -> torch.Tensor:
    portfolio_return = (weight * lret).sum(axis=1) - 1
    mean_return = portfolio_return.mean() * torch.tensor(252)
    volatility  = portfolio_return.std() * torch.sqrt(torch.tensor(252))
    return -mean_return/volatility


def plot_prediction(ground_truth:torch.Tensor, z_mean:torch.Tensor, z_std:torch.Tensor):
    ground_truth, z_mean, z_std = ground_truth.numpy(), z_mean.numpy(), z_std.numpy()
    T, n = z_mean.shape[0], z_mean.shape[1]
    
    # build confidence interval (1sigma, 2sigma)
    sig1_pos, sig1_neg = z_mean + z_std, z_mean - z_std
    sig2_pos, sig2_neg = z_mean + z_std * 2, z_mean - z_std * 2

    # start plotting
    fig, axs = plt.subplots(4,2, figsize=(20,15))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i == n:
            break
        ax.plot(np.arange(T), ground_truth[:,i], label='ground truth', color='black')
        ax.plot(np.arange(T), z_mean[:,i], ls='--', label='prediction', color='blue')
        ax.fill_between(np.arange(T), sig1_pos[:,i], sig1_neg[:,i], alpha=0.3, color='blue') # 1 sigma zones
        ax.fill_between(np.arange(T), sig2_pos[:,i], sig2_neg[:,i], alpha=0.15, color='purple') # 2 sigma zones

    plt.legend(['ground_truth', 'prediction'])
    plt.show()