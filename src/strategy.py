
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint


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
        
    def calc_weight(self, cov: np.ndarray, b:np.ndarray=None) -> np.array:
        # define initial weight to optimize
        if b is None:
            b = np.repeat([1/self.num_assets], self.num_assets) # 1/N
        init_w = np.array([1/self.num_assets] * self.num_assets)
        result = minimize(fun=self.obj_func, method='SLSQP', x0=init_w, args=(cov, b), bounds=self.bounds, constraints=self.constraint, options ={'ftol':1e-8}) # minimize port. variance
        return result['x'] #normalize the weight

    def run(self, enhanced:bool=False):
        """
        note that this strategy performs rebalancing once per month. (end of the month.)
        Hence we select monthly frequency.
        """
        # calculate portfolio weight first.
        lret = self.data['lret']
        months = pd.unique(lret.index.strftime('%Y-%m')) # a list containing values as 'yyyy-mm' format e.g '2019-01'
        b_table = self.adjust_b()

        for idx in range(1,len(months)):
            last_month = months[idx-1] 
            this_month = months[idx]

            # using the daily return data of the last month, update weight this month
            cov = lret.loc[last_month,:].cov().values
            if enhanced:
                b = b_table.loc[this_month,:].values
            else:
                b = None
            weight = self.calc_weight(cov, b)
            self.allocation.loc[this_month,:] = weight.T
        
    def performance(self):
        """
        run 메서드를 통해 계산한 weight 를 이용하여 포트폴리오의 성과를 계산합니다.
        """
        self.equity_curve = pd.DataFrame(index=self.data["price"].index, columns=['cumulative_return']) # cumulative return
        rv = self.data['rv'] 
        months = pd.unique(rv.index.strftime('%Y-%m')) # a list containing values as 'yyyy-mm' format e.g '2019-01'
        
        aum = 1 # for each month, we calculate daily return
        for idx in range(1,len(months)): 
            this_month = months[idx]
            self.equity_curve.loc[this_month,:] = (rv.loc[this_month,:].cumprod(axis = 0) @ self.allocation.loc[this_month,:].iloc[-1:].T).values * aum
            aum = self.equity_curve.loc[this_month,:].iloc[-1].values # update aum as the latest value of this month.

        df_profit_history = self.equity_curve
        df_profit_history = (df_profit_history / (df_profit_history.shift().fillna(1)))-1
        df_mean = df_profit_history.rolling(252).mean() * 252
        df_std = df_profit_history.rolling(252).std() * np.sqrt(252)
        self.annual_sharpe = (df_mean)/(df_std)