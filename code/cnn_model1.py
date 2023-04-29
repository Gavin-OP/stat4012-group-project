# Description: CNN 1D model1


import pandas as pd
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization, Activation, MaxPooling1D
from keras.optimizers import Adam
from keras.metrics import accuracy, mean_squared_error, Precision, Recall
from train_test_split import *
from normalization import *

# define model


def cnn_model1():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, padding='same',
              activation='relu', input_shape=(5, 7)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=2,
              padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(1))
    Activation('sigmoid')
    model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['accuracy'])
    return model


# fit model
model = cnn_model1()

random.seed(4012)
history = model.fit(X_train, y_train, epochs=100)
# plot history
plt.figure(figsize=(10, 8))
plt.plot(history.history['loss'], label='train')
plt.legend()
plt.show()

# make a prediction
y_pred = model.predict(X_test)
print(model.summary())

# plot prediction only display 5 x-axis label
plt.figure(figsize=(10, 8))
plt.plot(y_test.index, y_test, label='y_test')
plt.plot(y_test.index, y_pred, label='y_pred')
plt.xticks(y_test.index[::int(len(y_test) / 5)])
plt.legend()
plt.show()

# regenerate close price based on prediction return
y_pred_return = pd.Series(y_pred.flatten(), index=y_test.index)
# save y_pred_return to csv file
y_pred_return.to_csv('../data/y_pred_return.csv')

# close_pred = data['open'].iloc[int(
#     len(data) * 0.8):] * y_pred_return + data['open'].iloc[int(len(data) * 0.8):]
# close_test = data['open'].iloc[int(len(data) * 0.8):] * y_test + \
#     data['open'].iloc[int(len(data) * 0.8):]  

# close_day1 = data['close'].iloc[int(len(data) * 0.8):].shift(1)
# print('close_day1: ', close_day1)
# close_pred = []
# close_pred.append(close_day1 * y_pred_return.iloc[0] + close_day1)
# print('close_pred: ', close_pred)
# print('clength: y_pred_return', y_pred_return)
# # predict close price base on previous day close price
# for i in range(1, len(y_pred_return)):
#     close_pred.append(close_pred[i - 1] * y_pred_return.iloc[i] + close_pred[i - 1])
#     print('close_pred: ', close_pred[i])
    

close_test = data['close'].iloc[int(len(data) * 0.8):]

# plot data['close'] only display 5 x-axis label
# plt.figure(figsize=(10, 8))
# plt.plot(close_test, label='close')
# plt.plot(close_pred, label='close_pred')
# plt.xticks(close_test.index[::int(len(close_test) / 5)])
# # plt.plot(data['close'].iloc[int(len(data) * 0.8):].index, data['close'].iloc[int(len(data) * 0.8):] * y_pred_return, label='regenerated close')
# plt.legend()
# plt.show()
