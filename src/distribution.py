import numpy as np
from scipy.interpolate import interp1d


class ECDF_(object):
    """
    this is a class that implements empirical cumulative distribution of a data.
    Step function is used as a base for building the CDF, then linear interpolation is taken on these outputs.
    Also, this class has an inverse method that inputs y then outputs x. 
    """
    def __init__(self, data:np.ndarray):
        self.x = data.copy()
        self.x.sort()
        self.x_min, self.x_max = self.x[0], self.x[-1]
        self.n = len(self.x)
        self.y = np.linspace(1.0/self.n, 1.0, self.n)
        self.f = interp1d(self.x, self.y, fill_value='extrapolate') # make interpolation
        self.inv_f = interp1d(self.y, self.x, fill_value='extrapolate') # inverse is just arguments reversed
        
    def __call__(self, input_:np.ndarray) -> np.ndarray:
        """
        calculates y given x under defined ECDF class.
        """
        if np.sum(input_ > self.x_max) > 0 or np.sum(input_ < self.x_min) > 0:
            input_ = np.where(input_ > self.x_max, self.x_max, input_)
            input_ = np.where(input_ < self.x_min, self.x_min, input_)
            print(input_)
        return self.f(input_)

    def inverse(self, y:np.ndarray) -> np.ndarray:
        """
        calculates the inverse of ECDF with y as an input.
        """
         # if cdf value is less than 0 or more than 1, trim values
        if np.sum(y > 1.0) > 0 or np.sum(y < 0.0) > 0:
            y = np.where(y > 1.0, 1.0, y)
            y = np.where(y < 0.0, 0.0, y)
        return self.inv_f(y) # otherwise, return