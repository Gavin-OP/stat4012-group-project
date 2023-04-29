# Description:
# Split data into train and test set
# First 80% of data is train set, last 20% of data is test set
# Dimension:


import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
from feature_reshape import input_reshape
from feature_construction import PCA_feature_construction


def train_test_split_4012(X=0, y=0, n_days=5, stride=1, model='CNN', diff=False):
    if model == 'CNN':
        X, y = input_reshape(n_days, stride)
    elif model == 'LSTM':
        X, y = PCA_feature_construction(diff = False)
        X = array(X)
        y = array(y)
    
    # split data into train and test set by 80% and 20% by time
    X_train = X[:int(len(X) * 0.8)]
    X_test = X[int(len(X) * 0.8):]
    y_train = y[:int(len(y) * 0.8)]
    y_test = y[int(len(y) * 0.8):]

    print('X_train.shape:', X_train.shape)
    print('X_test.shape:', X_test.shape)
    print('y_train.shape:', y_train.shape)
    print('y_test.shape:', y_test.shape)

    return X_train, X_test, y_train, y_test


def plot_train_test_X(X_train, X_test):
    # plot the train and test features in one plot only display 5 x-axis label, make X_train and X_test as dataframe with continuous index, X_test first index is X_train last index + 1
    X_train = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
    X_test = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))
    X_train.index = range(len(X_train))
    X_test.index = range(len(X_train), len(X_train) + len(X_test))
    plt.figure(figsize=(10, 8))
    plt.plot(X_train[0], label='train')
    plt.plot(X_test[0], label='test')
    plt.legend()
    plt.show()


def plot_train_test_y(y_train, y_test):
    # plot the train and test label in one plot only display 5 x-axis label
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    y_train.index = range(len(y_train))
    y_test.index = range(len(y_train), len(y_train) + len(y_test))
    plt.figure(figsize=(10, 8))
    plt.plot(y_train.index, y_train, label='train')
    plt.plot(y_test.index, y_test, label='test')
    plt.xticks(y.index[::int(len(y) / 5)])
    plt.legend()
    plt.show()


X_train, X_test, y_train, y_test = train_test_split_4012(n_days=20, stride=1, model='LSTM', diff=False)
