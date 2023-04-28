# Description:
# Split data into train and test set
# First 80% of data is train set, last 20% of data is test set
# Dimension:
# X_train: (1077, 5, 7)
# X_test: (270, 5, 7)
# y_train: (1077, )
# y_test: (270, )


import pandas as pd
import matplotlib.pyplot as plt
from feature_reshape import *

# split data into train and test set by 80% and 20% by time
X_train = X[:int(len(X) * 0.8)]
X_test = X[int(len(X) * 0.8):]
y_train = y[:int(len(y) * 0.8)]
y_test = y[int(len(y) * 0.8):]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# plot the train and test label in one plot only display 5 x-axis label
plt.figure(figsize=(10, 8))
plt.plot(y_train.index, y_train, label='train')
plt.plot(y_test.index, y_test, label='test')
plt.xticks(y.index[::int(len(y) / 5)])
plt.legend()
plt.show()
