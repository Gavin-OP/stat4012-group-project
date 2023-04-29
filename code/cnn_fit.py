import random
from matplotlib import pyplot as plt
from cnn_model import cnn_model1
from train_test_split import train_test_split_4012

# fit model
def fit_model(seed=4012, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False):
    if cnn == 1:
        model = cnn_model1()

    X_train, X_test, y_train, y_test = train_test_split_4012(
        n_days=n_days, stride=stride, model=model_type, diff=diff)
    random.seed(seed)
    history = model.fit(X_train, y_train, epochs=epochs)
    # plot history
    plt.figure(figsize=(10, 8))
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.show()

    # save model, name it combined cnn, diff, n_days, stride
    filename = '../model/cnn_model' + str(cnn) + 'seed' + str(seed) + '_epochs' + str(epochs) +\
        '_days' + str(n_days) + '_stride' + str(stride) + \
        '_diff' + str(diff) + '.h5'
    model.save(filename)


fit_model(seed=4012, epochs=100, cnn=1, n_days=5,
          stride=1, model_type='CNN', diff=False)
