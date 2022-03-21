import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger('GPCopula.Data')
from torch.utils.data import Dataset

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
    def __init__(self, feature:pd.DataFrame, context_len:int=18, pred_len:int=4):     
        self.raw     = feature.iloc[:-1]
        self.raw_lbl = feature.shift(-1).iloc[context_len:-1].values
        
        self.data    = np.array([self.raw[i-context_len:i].values for i in range(context_len,self.raw.shape[0])])[:-pred_len]
        self.labels  = np.array([self.raw[i-pred_len:i].values for i in range(context_len+pred_len,self.raw.shape[0])])
        self.train_len = self.data.shape[0]

        assert self.data.shape[0] == self.labels.shape[0], 'not equal'
        logger.info(f'eval_len: {self.train_len}')

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return self.data[index], self.labels[index]