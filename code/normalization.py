# Description:
# Normalize features data exclude golden cross and death cross
# Using Min-Max Normalization

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_data():
    # define data
    data = pd.read_csv('../data/data.csv', header=0, index_col=0)

    # normalize features using min-max normalization
    scaler = MinMaxScaler()
    data[['open', 'high', 'low', 'close', 'daily_trading_volume',
          'RSI_14', 'BollW', 'precent_B', 'BIAS', 'ROC_1']] = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'daily_trading_volume',
                                                                                         'RSI_14', 'BollW', 'precent_B', 'BIAS', 'ROC_1']])

    # export Min and Max value of each feature to csv file
    column_index = [['open', 'high', 'low', 'close', 'daily_trading_volume',
                    'RSI_14', 'BollW', 'precent_B', 'BIAS', 'ROC_1']]
    scaler_min_max_data = pd.DataFrame(
        scaler.data_max_, index=column_index, columns=['Max'])
    scaler_min_max_data['Min'] = scaler.data_min_
    scaler_min_max_data.to_csv('../data/scaler.csv')
    return data
