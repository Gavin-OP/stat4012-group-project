import random
import numpy as np
from matplotlib import pyplot as plt
from cnn_model import cnn_model1, cnn_model2, cnn_model3, cnn_model4
from train_test_split import train_test_split_4012
import tensorflow as tf
from model_evaluate import predict_price

# fit model
def fit_model(seed=4012, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False):
    if cnn == 1:
        model = cnn_model1(n_days)
    elif cnn == 2:
        model = cnn_model2(n_days)
    elif cnn == 3:
        model = cnn_model3(n_days)
    elif cnn == 4:
        model = cnn_model4(n_days)

    X_train, X_test, y_train, y_test = train_test_split_4012(
        n_days=n_days, stride=stride, model=model_type, diff=diff)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    history = model.fit(X_train, y_train, epochs=epochs)
    # plot history
    plt.figure(figsize=(10, 8))
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    # plt.show()

    # save model, name it combined cnn, diff, n_days, stride
    filename = '../model/cnn_model' + str(cnn) + '_seed' + str(seed) + '_epochs' + str(epochs) +\
        '_days' + str(n_days) + '_stride' + str(stride) + \
        '_diff' + str(diff) + '.h5'
    model.save(filename)

    # evaluate model
    y_test, y_pred = predict_price(seed=seed, epochs=epochs, cnn=cnn, n_days=n_days, stride=stride, model_type=model_type, diff=diff)


if __name__ == "__main__":
    # fit_model(seed=4012, epochs=100, cnn=1, n_days=5,
    #       stride=1, model_type='CNN', diff=False)
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
    # fit_model(seed=777, epochs=100, cnn=2, n_days=10, stride=1, model_type='CNN', diff=False)
    # fit_model(seed=888, epochs=100, cnn=2, n_days=10, stride=1, model_type='CNN', diff=False)
    # fit_model(seed=988, epochs=100, cnn=2, n_days=10, stride=1, model_type='CNN', diff=False)
    # fit_model(seed=1988, epochs=100, cnn=2, n_days=10, stride=1, model_type='CNN', diff=False)
    # fit_model(seed=88, epochs=100, cnn=2, n_days=10, stride=1, model_type='CNN', diff=False)
    # fit_model(seed=4012, epochs=100, cnn=2, n_days=10, stride=1, model_type='CNN', diff=False)
    for i in range(34, 100):
        fit_model(seed=i, epochs=100, cnn=2, n_days=5, stride=1, model_type='CNN', diff=True)
        fit_model(seed=i, epochs=100, cnn=3, n_days=5, stride=1, model_type='CNN', diff=True)
        fit_model(seed=i, epochs=100, cnn=4, n_days=5, stride=1, model_type='CNN', diff=False)
    print('fit model done')
    
