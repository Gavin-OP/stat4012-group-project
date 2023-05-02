from keras.layers import LSTM,TimeDistributed,Dense,Dropout
from attention import Attention
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization, Activation, MaxPooling1D
from keras.optimizers import Adam
from keras.metrics import accuracy, mean_squared_error, Precision, Recall
from train_test_split import train_test_split_4012
import tensorflow as tf

# this script is used to generate LSTM models and save into h5 file

TIME_STEP = 5
FEATURE_NUM = 5

seed = 619
epochs = 100
model_num = 7
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# load data
X_train, X_test, y_train, y_test = train_test_split_4012(model='LSTM', diff=False)

# build model: LSTM - Attention
model = Sequential()
Attention(name='attention_weight')
model.add(LSTM(32, input_shape=(TIME_STEP,FEATURE_NUM)))  # These two variables are defined in configurations.py
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))
model.compile(metrics=['accuracy'],loss='mean_squared_error', optimizer='adam')

# fit model and save as h5 file
history = model.fit(X_train, y_train, epochs=epochs, batch_size=1).history
model.save(f'../model/lstm_model{model_num}_seed{seed}.h5')

# print model & loss function
print(model.summary())
plt.plot(history['loss'], linewidth=2, label='Train')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig(f'../graph/loss_lstm_model{model_num}_seed{seed}.png', dpi=1200, bbox_inches='tight')
plt.show()

# make predictions
y_pred = model.predict(X_test)
print(y_pred.shape)
np.savetxt(f"../data/lstm_model{model_num}_seed{seed}.csv", y_pred, delimiter=',')

# model evaluation
loss,accuracy = model.evaluate(X_test,y_test)
print(loss)
print(accuracy)
