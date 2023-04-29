import random
import numpy as np
from matplotlib import pyplot as plt
from cnn_model import cnn_model1
from train_test_split import train_test_split_4012
import tensorflow as tf

# fit model
def fit_model(seed=4012, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False):
    if cnn == 1:
        model = cnn_model1(n_days)

    X_train, X_test, y_train, y_test = train_test_split_4012(
        n_days=n_days, stride=stride, model=model_type, diff=diff)
    random.seed(seed)
    np.random.seed(seed)

    # tf.set_random_seed(seed)
    history = model.fit(X_train, y_train, epochs=epochs)
    # plot history
    plt.figure(figsize=(10, 8))
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.show()

    # save model, name it combined cnn, diff, n_days, stride
    filename = '../model/cnn_model' + str(cnn) + '_seed' + str(seed) + '_epochs' + str(epochs) +\
        '_days' + str(n_days) + '_stride' + str(stride) + \
        '_diff' + str(diff) + '.h5'
    model.save(filename)

if __name__ == "__main__":
    fit_model(seed=4012, epochs=100, cnn=1, n_days=20,
          stride=1, model_type='CNN', diff=False)
    # fit_model(seed=808, epochs=100, cnn=1, n_days=5,
    #       stride=1, model_type='CNN', diff=False)
    # fit_model(seed=123, epochs=100, cnn=1, n_days=5,
    #         stride=1, model_type='CNN', diff=False)
    # fit_model(seed=1155, epochs=100, cnn=1, n_days=5,
    #         stride=1, model_type='CNN', diff=False)
    # fit_model(seed=1702, epochs=100, cnn=1, n_days=5,
    #         stride=1, model_type='CNN', diff=False)
    # fit_model(seed=721, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # fit_model(seed=2001, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # fit_model(seed=144, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # fit_model(seed=1024, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # fit_model(seed=777, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    print('fit model done')
    
