# customized packages
from keras.layers import LSTM,TimeDistributed,Dense,Dropout
from attention import Attention
from keras.models import load_model
from configurations import *

# global packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization, Activation, MaxPooling1D
from keras.optimizers import Adam
from keras.metrics import accuracy, mean_squared_error, Precision, Recall
from train_test_split import train_test_split_4012
from evaluate import price_pred_graph

seed = 4005
epochs = 100
model_num = 2
np.random.seed(seed)

# load data
X_train, X_test, y_train, y_test = train_test_split_4012(model='LSTM', diff=False)

# build model: LSTM - Attention - 32 - 1
model = Sequential()
model.add(LSTM(32, input_shape=(TIME_STEP,FEATURE_NUM)))  # These two variables are defined in configurations.py
Attention(name='attention_weight')
model.add(Dense(32))
model.add(Dense(1))
model.compile(metrics=['accuracy'],loss='mean_squared_error', optimizer='adam')

# fit model
# model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=1).history
model.save(f'../model/lstm_model{model_num}_seed{seed}.h5')

# print model & loss function
print(model.summary())
plt.plot(history['loss'], linewidth=2, label='Train')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# make predictions
# train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
# print(train_pred.shape)
print(y_pred.shape)

# total_pred = np.append(train_pred,test_pred)
# print(total_pred)
np.savetxt(f"../data/lstm_model{model_num}_seed{seed}.csv", y_pred, delimiter=',')


# model evaluation
(loss,accuracy) = model.evaluate(X_test,y_test)
print(loss)
print(accuracy)

# draw prediction graph
# return_pred = pd.read_csv('../data/lstm_model1_seed4012.csv')
# print(return_pred)

y_true = pd.read_csv('../data/data.csv',index_col=0)['close']
# y_true = close.iloc[-276:-6]
print(y_true)


price_pred = []
price_pred.append(y_true.iloc[0])

i = 1
while i<len(y_pred)-1:
    price_pred.append(price_pred[i-1]*(1+float(y_pred.iloc[i-1].values)))
    i+=1

print(price_pred)
plt.plot(y_true,label='close')
plt.plot(price_pred,label='pred')
plt.xticks(y_true.index[::int(len(y_true) / 5)])
plt.show()




