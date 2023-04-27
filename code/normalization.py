import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# define data
data = pd.read_csv('../data/data.csv', header=0, index_col=0)

# normalize features using min-max normalization
scaler = MinMaxScaler()
data[['open', 'high', 'low', 'close', 'daily_trading_volume',
      'RSI_14', 'BollW', 'precent_B', 'BIAS', 'ROC_1']] = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'daily_trading_volume',
                                                                                     'RSI_14', 'BollW', 'precent_B', 'BIAS', 'ROC_1']])
# print(data.head(10))

