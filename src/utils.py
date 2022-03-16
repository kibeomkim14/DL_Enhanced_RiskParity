import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import PARAM_PATH, DATA_PATH
from scipy.stats import norm
from network import GPCopulaRNN
from typing import Optional, Tuple
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from torch.distributions.multivariate_normal import MultivariateNormal


def train_test_split(data:pd.DataFrame, test:bool=True):
    # preprocess the data for training
    y = data.iloc[20:][['AGG_ret020','EEM_ret020','IAU_ret020','IEF_ret020','IEV_ret020','ITOT_ret020','IYR_ret020']]
    y.index = data.index[:-20]
    X = data.iloc[:-20] 

    if test:
        # split train and test data
        tr_X, tr_y = X.loc[:'2015-09-21'], y.loc[:'2015-09-21']
        te_X, te_y = X.loc['2015-09-22':], y.loc['2015-09-22':]

        # normalize the inputs
        mu, std = tr_X.mean(axis=0), tr_X.std(axis=0)
        tr_X, te_X = tr_X.sub(mu).div(std), te_X.sub(mu).div(std) 
        return tr_X, tr_y, te_X, te_y
    else:
        tr_X, tr_y = X, y
        mu, std = tr_X.mean(axis=0), tr_X.std(axis=0)
        tr_X = tr_X.sub(mu).div(std)
        return tr_X, tr_y

def load_data() -> Tuple[pd.DataFrame,pd.DataFrame]:
    # import feature 
    prices = pd.read_csv(DATA_PATH+'/prices.csv', index_col='Date')

    # preprocess weekly data
    prices.index = pd.to_datetime(prices.index)
    prices_m = prices.resample('M').last()
    return prices, prices_m

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

    for i in range(Z.size(1)):
        Z_i = Z[:,i].numpy()
        
        # estimate empirical CDF
        # only use m datapoint to estimate CDF.
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


def inv_transform(X:torch.Tensor, cdfs:dict) -> pd.DataFrame:
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


def train_test_split(data_monthly:pd.DataFrame, split_ratio:float=0.7) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,dict]:
    """
    학습 데이터와 시험 데이터를 나눠주는 함수입니다. split_ratio 를 이용해서 원하는 비율로 나눌 수 있습니다.
    splits a dataset into train and test sets given a split ratio.

    INPUT
        data_monthly: pd.DataFrame
            monthly price of assets
        split_ratio: float
            specifies the ratio of train set over the whole.
    RETURNS
        Z_tr, Z_te, X_tr, X_te: Tuple[torch.tensor...]
            returns train & test monthly return data, raw and transformed
        cdf: dict
            dictionary containing a number of empirical CDFs.    
    """
    # monthly return
    lret_m = np.log(data_monthly/data_monthly.shift(1)).fillna(0.0) 

    # convert the data to torch tensor
    split_idx = int(lret_m.shape[0] * split_ratio)
    Z_tr, Z_te = torch.Tensor(lret_m.iloc[:split_idx].values), torch.Tensor(lret_m.iloc[split_idx:].values)
    X_tr, cdf = transform(Z_tr)
    X_te, _   = transform(Z_te, cdf)
    return Z_tr, Z_te, X_tr, X_te, cdf


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
    for idx in np.random.randint(0,tr_idx - context_len, size=num_samples):
        sample_indices.append((torch.LongTensor(idx + np.arange(context_len)), \
                            torch.LongTensor(idx + np.arange(prediction_len)+ context_len)))
    return sample_indices


def generate_portfolio_inputs(feature:pd.DataFrame, target:pd.DataFrame, num_trials:int=60) -> pd.DataFrame:
    data  = torch.Tensor(feature.values.copy())
    label = torch.Tensor(target.values.copy())

    # load trained parameters
    model = GPCopulaRNN()
    net_params = torch.load(PARAM_PATH+'forecaster_params.json')['net_state_dict']
    model.load_state_dict(net_params)
    
    # MC Dropout 
    predictions = []
    for _ in range(num_trials):
        predictions.append(model(data).detach())
    predictions = torch.stack(predictions)
    
    # calculate bayesian uncertainty estimates
    exp_returns = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)
    error = (predictions - label).abs().mean(axis=0)

    # concatenate and append
    meta_features = torch.concat([exp_returns, uncertainty, error], axis=1)
    meta_features = pd.DataFrame(meta_features.numpy(), index=feature.index)

    # rename columns 
    meta_features.columns = [name.split('_')[0] + '_exp_return' for name in target.columns.values]  + \
                            [name.split('_')[0] + '_uncertainty' for name in target.columns.values] + \
                            [name.split('_')[0] + '_mae' for name in target.columns.values] 

    # calculate exponential average of mae features
    meta_features.iloc[:,-8:] = meta_features.iloc[:,-8:].ewm(10).mean()
    return meta_features


def gaussian_loss(x:torch.Tensor, mu_t:torch.Tensor, cov_t:torch.Tensor) -> torch.Tensor:
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
    return -loglikelihood.sum() # negative log-likelihood


def sharpe_ratio(weight:torch.Tensor, lret:torch.Tensor) -> torch.Tensor:
    portfolio_return = (weight * lret).sum(axis=1) - 1
    mean_return = portfolio_return.mean() * torch.tensor(252)
    volatility  = portfolio_return.std() * torch.sqrt(torch.tensor(252))
    return -mean_return/volatility



class ECDF_(object):
    def __init__(self, data:np.ndarray):
        self.x = data
        self.x.sort()
        self.x_min, self.x_max = self.x[0], self.x[-1]
        self.n = len(self.x)
        self.y = np.linspace(1.0/self.n, 1.0, self.n)
        self.f = interp1d(self.x, self.y, fill_value='extrapolate') # make interpolation
        self.inv_f = interp1d(self.y, self.x, fill_value='extrapolate') # inverse is just arguments reversed
        
    def __call__(self, x:np.ndarray) -> np.ndarray:
        """
        calculates y given x under defined ECDF class.
        """
        if np.sum(x > self.x_max) > 0 or np.sum(x < self.x_min) > 0:
            x = np.where(x > self.x_max, self.x_max, x)
            x = np.where(x < self.x_min, self.x_min, x)
        return self.f(x)

    def inverse(self, y:np.ndarray) -> np.ndarray:
        """
        calculates the inverse of ECDF with y as an input.
        """
         # if cdf value is less than 0 or more than 1, trim values
        if np.sum(y > 1.0) > 0 or np.sum(y < 0.0) > 0:
            y = np.where(y > 1.0, 1.0, y)
            y = np.where(y < 0.0, 0.0, y)
        return self.inv_f(y) # otherwise, return

