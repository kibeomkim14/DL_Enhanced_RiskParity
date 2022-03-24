from re import S
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from network import GPCopulaNet
from scipy.optimize import minimize, Bounds, LinearConstraint
from utils import inv_transform_3D


class RiskParity:
    def __init__(self, data:pd.DataFrame, bound:list = [0, np.inf]):
        self.data = {}
        self.data["price"] = data
        self.data["lret"]  = np.log(data/data.shift(1)).fillna(0)
        self.data["rv"]    = self.data["lret"].add(1)
        self.num_assets = self.data["price"].shape[1]

        # portoflio weight and performance dataframes which are to be filled.
        self.allocation   = pd.DataFrame(index=self.data["price"].index, columns=self.data["price"].columns)
        self.equity_curve = pd.DataFrame(index=self.data["price"].index, columns=['Risk_Parity']) # cumulative return
        
        # Optimization constraints
        self.bounds = Bounds([bound[0]]*self.num_assets, [bound[1]]*self.num_assets)  # 모든 weight 의 lower limit 부여. Default: 0 <= x
        self.constraint = LinearConstraint([[1]*self.num_assets], [1], [1])  # 모든 weight 의 합은 1 이라는 조건 부여
  
    def obj_func(self, w: np.ndarray, cov: np.ndarray, b:np.ndarray):
        sigma = np.sqrt(w.T @ cov @ w)
        x = w/sigma
        return np.sum(abs((cov @ x) * x - b))
        
    def calc_weight(self, cov: np.ndarray, b:Optional[np.ndarray]=None) -> np.array:
        # define initial weight to optimize
        if b is None:
            b = np.repeat([1/self.num_assets], self.num_assets) # 1/N
        init_w = np.array([1/self.num_assets] * self.num_assets)
        result = minimize(fun=self.obj_func, method='SLSQP', x0=init_w, args=(cov, b), bounds=self.bounds, constraints=self.constraint, options ={'ftol':1e-8}) # minimize port. variance
        return result['x'] #normalize the weight

    def run(self):
        """
        note that this strategy performs rebalancing once per month. (end of the month.)
        Hence we select monthly frequency.
        """
        # calculate portfolio weight first.
        lret = self.data['lret']
        months = pd.unique(lret.index.strftime('%Y-%m')) # a list containing values as 'yyyy-mm' format e.g '2019-01'

        for idx in range(6,len(months)):
            start_month = months[idx-6] 
            end_month   = months[idx-1] 
            this_month  = months[idx]

            # using the daily return data of the last month, update weight this month
            cov = lret.loc[start_month:end_month,:].cov().values
            weight = self.calc_weight(cov)
            self.allocation.loc[this_month,:] = weight.T
        
    def performance(self):
        """
        run 메서드를 통해 계산한 weight 를 이용하여 포트폴리오의 성과를 계산합니다.
        """
        self.equity_curve = pd.DataFrame(index=self.data["price"].index, columns=['cumulative_return']) # cumulative return
        rv = self.data['rv'] 
        months = pd.unique(rv.index.strftime('%Y-%m')) # a list containing values as 'yyyy-mm' format e.g '2019-01'
        
        aum = 1 # for each month, we calculate daily return
        for idx in range(6,len(months)): 
            this_month = months[idx]
            self.equity_curve.loc[this_month,:] = (rv.loc[this_month,:].cumprod(axis = 0) @ self.allocation.loc[this_month,:].iloc[-1:].T).values * aum
            aum = self.equity_curve.loc[this_month,:].iloc[-1].values # update aum as the latest value of this month.
            print(aum, idx)

        df_profit_history = self.equity_curve
        df_profit_history = (df_profit_history / (df_profit_history.shift().fillna(1)))-1
        df_mean = df_profit_history.rolling(252).mean() * 252
        df_std = df_profit_history.rolling(252).std() * np.sqrt(252)
        self.annual_sharpe = (df_mean)/(df_std)


class EnhancedRiskParity(RiskParity):
    def __init__(self, model:nn.Module, data:pd.DataFrame, data_w:pd.DataFrame, cdf:dict, bound:list = [0, np.inf]):
        super().__init__(data, bound)
        self.data["price_w"] = data_w
        self.data["lret_w"]  = np.log(data_w/data_w.shift(1)).fillna(0)
        self.b_table = pd.DataFrame(index=self.data["price"].index, columns=self.data["price"].columns)
        self.prediction_mean = pd.DataFrame(index=self.data["price"].index, columns=self.data["price"].columns)
        self.prediction_uncertainty = pd.DataFrame(index=self.data["price"].index, columns=self.data["price"].columns)
        
        self.cdf   = cdf
        self.model = model
    
    def run(self, verbose:int=0):
        """
        note that this strategy performs rebalancing once per month. (end of the month.)
        Hence we select monthly frequency.
        """
        # calculate portfolio weight first.
        lret_d = self.data['lret']
        lret_w = self.data['lret_w']
        months = pd.unique(lret_w.index.strftime('%Y-%m')) # a list containing values as 'yyyy-mm' format e.g '2019-01'
        
        for idx in range(6,len(months)):
            start_month = months[idx-6]
            end_month   = months[idx-1]
            this_month  = months[idx]

            # take 6 months of data, estimate covariance with 6 months daily return data
            data_d = lret_d.loc[start_month:end_month,:]
            data_w = lret_w.loc[start_month:end_month,:]
            cov_past = data_d.cov().values

            # adjust risk target given model prediction
            # calculate weight under Risk Contribution scheme given target b.
            b, mu, std  = self.kelly_b(data_w, lret_w.loc[this_month])
            if verbose == 1:
                print(f'time:{this_month}, target:{b}')
            weight = self.calc_weight(cov_past, b)
            
            # store in the table
            self.b_table.loc[this_month,:]    = b.T 
            self.allocation.loc[this_month,:] = weight.T
            self.prediction_mean.loc[this_month,:] = mu
            self.prediction_uncertainty.loc[this_month,:] = std

    def kelly_b(
            self,
            data:pd.DataFrame, 
            targ:pd.DataFrame, 
            return_target:float=0.03,
            cap:list=[0.1, 0.0] 
            ):        
        """
        Main idea taken from the web: https://en.wikipedia.org/wiki/Kelly_criterion/Investment_Formula
        """        
        self.model.eval()
        risk_target = np.tile([1/data.shape[1]], data.shape[1])
        Z, targ = torch.Tensor(data.values), torch.Tensor(targ.values)

        # pass the past inputs then predict the next value 
        _, _    = self.model(torch.Tensor(Z[-2:-1]),torch.arange(self.b_table.shape[1]))
        x_pred  = self.model.predict(torch.Tensor(Z[-2:-1]), num_samples=20, pred_len=targ.size(0))
        z_pred  = inv_transform_3D(x_pred, self.cdf)

        z_pred_m = (z_pred + 1).prod(axis=1) - 1 # monthly return prediction
        mu, std  = z_pred_m.mean(axis=0), z_pred_m.std(axis=0)

        # Kelly Approach
        kelly_value = []
        for i in range(mu.size(0)):
            # define distribution of prediction samples
            Nd = torch.distributions.Normal(loc=mu[i], scale=std[i])
            samples = Nd.sample((50,))

            # find winning/losing prob and conditional expectation of winning/losing
            q = Nd.cdf(torch.Tensor([return_target])) # losing prob, P(Z < 0)
            p = 1 - q # winning prob
            b = samples[samples >= return_target].mean()
            a = samples[samples < return_target].mean()

            # calculate capped kelly crietrion
            kelly = p/a.abs() - q/b
            kelly_value.append(kelly.item()/100)
        kelly_value = np.array(kelly_value)    
        kelly_value = np.where(kelly_value > cap[0], cap[0], kelly_value)
        kelly_value = np.where(kelly_value < cap[1], cap[1], kelly_value)
        b = kelly_value + risk_target
        return b/b.sum(), mu.numpy(), std.numpy()