import numpy as np
import pandas as pd

def train_test_split(data:pd.DataFrame, valid_ratio:float=0.2, train_ratio:float=0.6):
    # we will use 60% of the data as train+val set. this data will be used to tune DL model.
    n = data.shape[0]
    tr_idx  = int(n * train_ratio)
    val_idx = int(tr_idx * (1-valid_ratio))

    # split the dataset according to tr index
    tr_data = data.iloc[:tr_idx]
    te_data = data.iloc[tr_idx:]

    # Now we create a target for training a DL model. Target will be a future 130-days return of each ETF products.
    # To make a target we look 130 days ahead of the training set. This will be done by shifting the view by 130 days.
    target_list = ['BNDX_ret130','BND_ret130','VGK_ret130','VNQI_ret130',
                    'VNQ_ret130','VTI_ret130','VWOB_ret130','VWO_ret130']
    target = data.iloc[130:tr_idx+130][target_list]

    # add prefix on columns and reset the
    target = target.add_prefix('targ_')
    target.index = tr_data.index

    assert tr_data.shape[0] == target.shape[0], 'feature shape and target shape is not same.'
    mu, std = tr_data.mean(axis=0), tr_data.std(axis=0)
    tr_data = tr_data.sub(mu).div(std) # normalize the data along rows

    trn_features, trn_label = tr_data.iloc[:val_idx-130], target.iloc[:val_idx-130] # cut last few days of data to prevent data leakage.
    val_features, val_label = tr_data.iloc[val_idx:], target.iloc[val_idx:]

    return trn_features, trn_label, val_features, val_label, te_data