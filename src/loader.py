import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger('GPCopula.Data')


class TrainDataset(Dataset):

    def __init__(self, feature:pd.DataFrame, context_len:int=18):
        T = len(pd.unique(feature.index))
        
        # sequentialize the data by asset id
        data, labels = [], []
        for idx in pd.unique(feature.asset_id): 
            for t in range(context_len, T):
                datum = feature[feature.asset_id == idx].iloc[t-context_len:t, :-1].values
                label = feature[feature.asset_id == idx].iloc[t-1].label
                data.append(datum)
                labels.append(label)
        self.data  = np.array(data)
        self.label = labels
        self.train_len = self.data.shape[0]
        logger.info(f'train_len: {self.train_len}')

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1], int(self.data[index,0,-1]), self.label[index])