# Description: naive LSTM model

from keras.layers import LSTM,TimeDistributed,Dense,Dropout
from keras.models import load_model

import pandas as pd
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization, Activation, MaxPooling1D
from keras.optimizers import Adam
from keras.metrics import accuracy, mean_squared_error, Precision, Recall
from feature_construction import PCA_feature_construction


# define model
BATCH_START = 0
TIME_STEPS = 5
BATCH_SIZE = 32
INPUT_SIZE = 25
OUTPUT_SIZE = 4
PRED_SIZE = 15 #预测输出1天序列数据
CELL_SIZE = 128
LR = 0.0001
EPOSE = 1000

# fit model
class KerasMultiLSTM(object):

    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size  # LSTM神经单元数
        self.batch_size = batch_size  # 输入batch_size大小

    def model(self):
        self.model = Sequential()

        # 不固定batch_size，预测时可以以1条记录进行分析
        self.model.add(LSTM(units=self.cell_size, activation='relu', return_sequences=True,
                            input_shape=(self.n_steps, self.input_size))
                       )
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=self.cell_size, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=self.cell_size, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))

        # 全连接，输出， add output layer
        self.model.add(TimeDistributed(Dense(self.output_size)))
        self.model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
        self.model.summary()

    def train(self, x_train, y_train, epochs, filename):
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=self.batch_size).history
        self.model.save(filename)
        return history

if __name__ == '__main__':
    random.seed(4012)

    X, y = PCA_feature_construction(diff=False)
    X_train = X[:int(len(X) * 0.8)]
    X_test = X[int(len(X) * 0.8):]
    y_train = y[:int(len(y) * 0.8)]
    y_test = y[int(len(y) * 0.8):]

    # training data's length needs to be a multiple of batch size
    k = len(X_train) % BATCH_SIZE # k is the remainder
    X_train, y_train = X_train[k:], y_train[k:]
    print(f"Length of new training data set is: {len(X_train)}")

    model = KerasMultiLSTM(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    model.model()
    history = model.train(train_x, train_y, EPOSE, "lstm-model2.h5")

    plt.plot(history['loss'], linewidth=2, label='Train')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()



