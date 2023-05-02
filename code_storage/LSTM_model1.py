# Description: naive LSTM model

from keras.layers import LSTM,TimeDistributed,Dense,Dropout
from attention import Attention
from keras.models import load_model

import pandas as pd
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization, Activation, MaxPooling1D
from keras.optimizers import Adam
from keras.metrics import accuracy, mean_squared_error, Precision, Recall
from train_test_split import train_test_split_4012
from evaluate import price_pred_graph

# define model
BATCH_START = 0    # bacth起始点a
TIME_STEPS = 5     # 时间跨度
BATCH_SIZE = 1    # 每次喂进model的sample数
INPUT_SIZE = 7     # feature数
# PRED_SIZE = 15 #预测输出1天序列数据
CELL_SIZE = 32    # LSTM 神经元数
LR = 0.0001
EPOSE = 100

# fit model
class KerasMultiLSTM(object):

    def __init__(self, n_steps, input_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        # self.output_size = output_size
        self.cell_size = cell_size  # LSTM神经单元数
        self.batch_size = batch_size  # 输入batch_size大小

    def model(self):
        self.model = Sequential()
        # 不固定batch_size，预测时可以以1条记录进行分析
        # attention_mul = attention_3d_block(inputs)
        self.model.add(LSTM(units=self.cell_size, activation='relu', return_sequences=True,
                            input_shape=(self.n_steps, self.input_size)))

        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=self.cell_size, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))
        # self.model.add(LSTM(units=self.cell_size, activation='relu', return_sequences=True))
        # self.model.add(Dropout(0.2))

        # 全连接，输出， add output layer
        self.model.add(TimeDistributed(Dense(16)))
        self.model.add(TimeDistributed(Dense(1)))
        self.model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
        self.model.summary()

    def train(self, x_train, y_train, epochs, filename):
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=self.batch_size).history
        self.model.save(filename)
        return history

if __name__ == '__main__':
    random.seed(4012)
    X_train, X_test, y_train, y_test = train_test_split_4012(model='LSTM', diff=False)

    # training data's length needs to be a multiple of batch size
    k = len(X_train) % BATCH_SIZE # k is the remainder
    X_train, y_train = X_train[k:], y_train[k:]
    print(f"Length of new training data set is: {len(X_train)}")

    model = KerasMultiLSTM(TIME_STEPS, INPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    model.model()
    history = model.train(X_train, y_train, EPOSE,'lstm1.h5')

    plt.plot(history['loss'], linewidth=2, label='Train')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    y_pred = model.predict(X_test)
    y_pred.to_csv('y_pred_return_lstm.csv')

