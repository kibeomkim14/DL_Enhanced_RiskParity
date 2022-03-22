import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger('GPCopula.Data')

class TrainDataset(Dataset):
    def __init__(self, feature:pd.DataFrame, context_len:int=18):     
        self.raw    = feature.iloc[:-1]
        self.data   = np.array([self.raw[i-context_len:i].values for i in range(context_len,self.raw.shape[0])])
        self.labels = feature.shift(-1).iloc[context_len:-1].values
        self.train_len = self.data.shape[0]

        assert self.data.shape[0] == self.labels.shape[0], 'not equal'
        logger.info(f'train_len: {self.train_len}')

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return self.data[index], self.labels[index] 


class EvalDataset(Dataset):
    def __init__(self, feature:pd.DataFrame, context_mth:int=6, pred_mth:int=1):  
        
        self.context_mth = context_mth
        self.pred_mth    = pred_mth
        self.months      = pd.unique(feature.index.strftime('%Y-%m')) 
        self.data        = np.array([feature.loc[self.months[idx-context_mth]:self.months[idx-1]].values for idx in range(context_mth, self.months.shape[0])], dtype='object')
        self.labels      = np.array([feature.loc[self.months[idx-1+pred_mth]].values for idx in range(context_mth, self.months.shape[0])], dtype='object')
        self.train_len   = self.data.shape[0]
        assert self.data.shape[0] == self.labels.shape[0], 'not equal'

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.months[index+self.context_mth]