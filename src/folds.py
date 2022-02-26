import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

class PurgedKFold(_BaseKFold):
    """
    출처: Advances in Financial Machine Learning (Marcos de Prado, 2018)
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between 
    """
    def __init__(self, n_splits=5, t1=None, pctEmbargo=0.1):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None) 
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y = None, groups = None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index') 
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        fold_boundary = [(fold[0], fold[-1]+1) for fold in np.array_split(np.arange(X.shape[0]), self.n_splits)] 
        
        for start, end in fold_boundary:
            t0 = self.t1.index[start] # start of test set
            test_indices = indices[start:end] # test indices
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max()) 
            train_indices = self.t1.index.searchsorted(self.t1[self.t1<=t0].index) 
            if maxT1Idx < X.shape[0]: # right train (with embargo)
                train_indices = np.concatenate((train_indices,indices[maxT1Idx+mbrg:])) 
            yield train_indices, test_indices