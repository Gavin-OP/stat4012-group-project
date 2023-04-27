# Description:
# This file is used to clean the raw data and generate the data.csv file.
# data.csv file contains 12 features and 1 label. 
# Features include open, high, low, close, daily_trading_volume, RSI_14, BollW, precent_B, BIAS, ROC_1, golden_cross and death_cross.
# The golden_cross and death_cross are calculated by the MA_5 and MA_25.
# Label is the sixth day's return.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('../data/raw_data_adjusted.xlsx',
                     index_col=0).dropna(how='any')
data.sort_values(by='date', ascending=True, inplace=True)

# calculate golden and death cross
data['MA_5'] = data['close'].rolling(5).mean()
data['MA_25'] = data['close'].rolling(25).mean()
data['diff'] = np.sign(data["MA_5"] - data["MA_25"])
data['signal'] = np.sign(data['diff'] - data['diff'].shift(1))
data['golden_cross'] = data['signal'].map({1: 1, 0: 0, -1: 0})
data['death_cross'] = data['signal'].map({-1: 1, 0: 0, 1: 0})
data = data.drop(columns=['diff', 'signal']).dropna(how='any')
print(data[['golden_cross', 'death_cross']].sum())

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

X = data[['open', 'high', 'low', 'close', 'daily_trading_volume',
          'RSI_14', 'BollW', 'precent_B', 'BIAS', 'ROC_1', 'golden_cross', 'death_cross']]

# calculate the sixth day's return
X['sixth_day_return'] = (
    (X['close'].shift(-5)-X['open'].shift(-5))/X['open'].shift(-5)).dropna(how='any')
print(X.head(10))

# store X as csv file
X.to_csv('../data/data.csv')
