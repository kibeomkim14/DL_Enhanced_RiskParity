import numpy as np
import pandas as pd

from config import DATA_PATH

def load_data(interval:str='M', calc_ret=False) -> pd.DataFrame:
    # import feature 
    data = pd.read_csv(DATA_PATH+'/prices.csv', index_col='Date')

    # preprocess weekly data
    data.index = pd.to_datetime(data.index)
    if interval != 'D':
        data = data.resample(interval).last()
    
    if calc_ret:
        data = np.log(data/data.shift(1)).fillna(0.0)
    return data

def preprocess(df:pd.DataFrame) -> pd.DataFrame:
    asset_list = df.columns.values

    # create weekly, monthly, quarterly return
    l_ret = np.log(df_price/df_price.shift()).fillna(0.0)
    l_ret = l_ret.add_suffix('_w_ret')
    l_ret_m = np.log(df_price/df_price.shift(4)).fillna(0.0)
    l_ret_m = l_ret_m.add_suffix('_m_ret')
    l_ret_q = np.log(df_price/df_price.shift(12)).fillna(0.0)
    l_ret_q = l_ret_q.add_suffix('_q_ret')

    #concatenate features
    feature = pd.concat([l_ret, l_ret_m, l_ret_q], axis=1)
    label   = feature.iloc[1:,1:8]
    feature = feature.iloc[:-1]
    feature, label = feature.iloc[12:], label.iloc[12:]
    feature = feature.reset_index()

    df_all = pd.DataFrame()
    for i, asset in enumerate(asset_list):
        df = feature.filter(like=asset, axis=1).copy()
        df.columns = ['weekly_return', 'monthly_return', 'quarterly_return']
        df['asset_id'] = i
        df['date'] = feature['Date']
        df_all = pd.concat([df_all, df], axis=0)

    df_all['label'] = df_all.groupby('asset_id')['weekly_return'].shift(-1) 
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.dropna()    
    df_all = df_all.sort_values(by=['date','asset_id'])
    return feature

if __name__ == "__main__":
    df_price = load_data('W', True) # weekly prices
    features = preprocess(df_price)
    print('Preprocessing Complete. Now saving the file...')
    df_price.to_csv('/Users/mac/Desktop/PycharmProjects/TAADL/input/log_return.csv')
    features.to_csv('/Users/mac/Desktop/PycharmProjects/TAADL/input/features.csv')
    print('Saved.')