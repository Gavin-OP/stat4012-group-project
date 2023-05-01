import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from train_test_split import train_test_split_4012
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def predict_price(seed=4012, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False, epochs=100, model_num=0, good = 'NO'):
    if model_type == 'CNN':
        if good == 'NO':
            modelname = '../model/cnn_model' + str(cnn) + '_seed' + str(seed) + '_epochs' + str(epochs) +\
                    '_days' + str(n_days) + '_stride' + str(stride) + \
                    '_diff' + str(diff) + '.h5'
        elif good == 'good':
            modelname = '../model/cnn_model' + str(cnn) + '_seed' + str(seed) + '_epochs' + str(epochs) +\
            '_days' + str(n_days) + '_stride' + str(stride) + \
            '_diff' + str(diff) + '_good.h5'
    elif model_type == 'LSTM':
        surname = f'lstm_model{model_num}_seed{seed}'
        modelname = '../model/' + surname + '.h5'

    model = load_model(modelname)
    X_train, X_test, y_train, y_test = train_test_split_4012(
        n_days=n_days, stride=stride, model=model_type, diff=diff)

    # make a prediction
    y_pred = model.predict(X_test)  # y_pred is the predicted return
    print(model.summary())

    # save result
    if model_type == 'LSTM':
        np.savetxt(f'../prediction/{surname}.csv',y_pred)
    elif model_type == 'CNN':
        if good == 'good':
            np.savetxt(f'../prediction/cnn_model{cnn}_seed{seed}_epochs{epochs}_days{n_days}_stride{stride}_diff{diff}_good.csv',y_pred)

    # return_pred_plot(y_test, y_pred)
    price_pred_graph(y_pred, seed=seed, cnn=cnn, n_days=n_days,
                     stride=stride, model_type=model_type, diff=diff, epochs=epochs)
    return y_test, y_pred

# plot prediction only display 5 x-axis label


def return_pred_plot(y_test, y_pred):
    y_test = pd.DataFrame(y_test)
    y_pred = pd.DataFrame(y_pred)
    y_test.index = range(len(y_test))
    y_pred.index = range(len(y_test))
    plt.figure(figsize=(10, 8))
    plt.plot(y_test.index, y_test, label='y_test')
    plt.plot(y_test.index, y_pred, label='y_pred')
    plt.xticks(y_test.index[::int(len(y_test) / 5)])
    plt.legend()
    plt.show()


def price_pred_graph(return_pred, seed=4012, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False, epochs=100):
    return_pred = pd.DataFrame(return_pred)
    close = pd.read_csv('../data/data.csv', index_col=0)['close']
    y_true = close.iloc[-276:-6]

    y_pred = []
    y_pred.append(y_true.iloc[0])

    i = 1
    while i < len(return_pred)-1:
        y_pred.append(y_pred[i-1]*(1+float(return_pred.iloc[i-1].values)))
        i += 1

    plt.figure(figsize=(10, 8))
    plt.plot(y_true, label='close')
    plt.plot(y_pred, label='pred')
    plt.xticks(y_true.index[::int(len(y_true) / 5)])
    plt.legend()
    filename = '../graph/cnn_model' + str(cnn) + '_price_pred_seed' + str(seed) + '_epochs' + str(epochs) +\
        '_days' + str(n_days) + '_stride' + str(stride) + \
        '_diff' + str(diff) + '.png'
    plt.savefig(filename, dpi=1200, bbox_inches='tight')
    # plt.show()

def CNN_classification(seed=4012, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False, epochs=100, model_num=0, good = 'NO'):
    y_test, y_pred = predict_price(seed=seed, cnn=cnn, n_days=n_days, stride=stride, model_type=model_type, diff=diff, epochs=epochs, good=good)
    y_pred_prob = 1 / (1 + np.exp(-y_pred))

    y_pred_class = np.where(y_pred_prob > 0.5, 1, 0)
    y_test_class = np.where(y_test > 0, 1, 0)

    print('Accuracy: ', accuracy_score(y_test_class, y_pred_class))
    print('Precision: ', precision_score(y_test_class, y_pred_class))
    print('Recall: ', recall_score(y_test_class, y_pred_class))
    print('F1: ', f1_score(y_test_class, y_pred_class))
    print('Confusion Matirx: ', confusion_matrix(y_test_class, y_pred_class))

    # plot roc curve
    fpr, tpr, thresholds = roc_curve(y_test_class, y_pred_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()





if __name__ == "__main__":
    # y_test, y_pred = predict_price(seed = 4012, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # y_test, y_pred = predict_price(seed = 808, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # y_test, y_pred = predict_price(
    #     seed=123, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # y_test, y_pred = predict_price(seed=1155, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # y_test, y_pred = predict_price(seed=1702, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # y_test, y_pred = predict_price(seed=721, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # y_test, y_pred = predict_price(seed=2001, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # y_test, y_pred = predict_price(seed=144, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # y_test, y_pred = predict_price(seed=1024, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # y_test, y_pred = predict_price(seed=777, epochs=100, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False)
    # y_test, y_pred = predict_price(seed=998, epochs=100, cnn=2, n_days=10, stride=1, model_type='CNN', diff=False, good='good')
    y_test, y_pred = predict_price(seed=998, epochs=100, cnn=2, n_days=5, stride=1, model_type='CNN', diff=False, good='good')
    y_test, y_pred = predict_price(seed=61, epochs=100, cnn=4, n_days=5, stride=1, model_type='CNN', diff=True, good='good')
