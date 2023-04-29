# Description:
# This file is used to clean the raw data and generate the data.csv file.
# data.csv file contains 12 features and 1 label. 
# Features include open, high, low, close, daily_trading_volume, RSI_14, BollW, precent_B, BIAS, ROC_1, golden_cross and death_cross.
# The golden_cross and death_cross are calculated by the MA_5 and MA_25.
# Label is the sixth day's return.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_data = pd.read_excel('../data/raw_data_adjusted.xlsx',
                     index_col=0).dropna(how='any')
raw_data.sort_values(by='date', ascending=True, inplace=True)

# add golden cross and death cross
def golden_death_cross_calculation(data):
    # calculate golden and death cross
    data['MA_5'] = data['close'].rolling(5).mean()
    data['MA_25'] = data['close'].rolling(25).mean()
    data['diff'] = np.sign(data["MA_5"] - data["MA_25"])
    data['signal'] = np.sign(data['diff'] - data['diff'].shift(1))
    data['golden_cross'] = data['signal'].map({1: 1, 0: 0, -1: 0})
    data['death_cross'] = data['signal'].map({-1: 1, 0: 0, 1: 0})

    # let golden cross and death cross NAN to be 0
    data['golden_cross'].fillna(0, inplace=True)
    data['death_cross'].fillna(0, inplace=True)
    data = data.drop(columns=['diff', 'signal'])
    print(data[['golden_cross', 'death_cross']].sum())
    return data

# plot the golden and death cross with MA_5 and MA_25
def golden_death_cross_plot(data):
    data = golden_death_cross_calculation(data)
    # plot the golden and death cross
    plt.figure(figsize=(10, 5))
    # plt.plot(data['close'], label='close')
    plt.plot(data['MA_5'], label='MA_5')
    plt.plot(data['MA_25'], label='MA_25')
    plt.scatter(data[data['golden_cross'] == 1].index,
                data[data['golden_cross'] == 1]['close'], label='golden_cross', marker='^', color='green')
    plt.scatter(data[data['death_cross'] == 1].index,
                data[data['death_cross'] == 1]['close'], label='death_cross', marker='v', color='red')
    plt.title('MA_5, MA_25 and stock price')
    plt.legend()
    plt.show()

data = golden_death_cross_calculation(raw_data)
X = data[['open', 'high', 'low', 'close', 'daily_trading_volume',
          'RSI_14', 'BollW', 'precent_B', 'BIAS', 'ROC_1', 'golden_cross', 'death_cross']]
print(type(X))
# calculate the sixth day's return
# X['sixth_day_return'] = (
#     (X['close'].shift(-5)-X['open'].shift(-5))/X['open'].shift(-5)).dropna(how='any')
X.loc[:, 'sixth_day_return'] = (X['close'].shift(-5) - X['close'].shift(-6)) /  X['close'].shift(-6).dropna(how='any')
print('Cleaned data is saved in data.csv. It contains column named:\n', data.columns)
print(X.head())

# store X as csv file
X.to_csv('../data/data.csv')
