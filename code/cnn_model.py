# Description: CNN 1D model1


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, BatchNormalization, Activation, MaxPooling1D
from tensorflow.keras.optimizers import Adam


def cnn_model1(n_days):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, padding='same',
              activation='relu', input_shape=(n_days, 7)))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=2,
              padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(1))
    # Activation('sigmoid')
    model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['accuracy'])
    return model

