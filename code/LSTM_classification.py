import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from train_test_split import train_test_split_4012
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os



def return_pred_plot(y_test, y_pred, model_num):
    y_test = pd.DataFrame(y_test)
    y_pred = pd.DataFrame(y_pred)
    y_test.index = range(len(y_test))
    y_pred.index = range(len(y_test))
    plt.figure(figsize=(10, 8))
    plt.plot(y_test.index, y_test, label='y_test')
    plt.plot(y_test.index, y_pred, label='y_pred')
    plt.xticks(y_test.index[::int(len(y_test) / 5)])
    plt.legend()
    filename = f'../graph/LSTM_graph/return_lstm_model{model_num}'
    plt.savefig(filename, dpi=1200, bbox_inches='tight')
    plt.show()


def price_pred_graph(return_pred, model_num):
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

    plt.plot(y_pred, label='prediction')
    plt.xticks(y_true.index[::int(len(y_true) / 5)])
    plt.legend()
    filename = f'../graph/LSTM_graph/price_lstm_model{model_num}'
    plt.savefig(filename, dpi=1200, bbox_inches='tight')
    plt.show()


def evaluate_regression(seed, epochs=100, model_num=0):
    surname = f'lstm_model{model_num}_seed{seed}'
    modelname = '../model/' + surname + '.h5'
    model = load_model(modelname)
    X_train, X_test, y_train, y_test = train_test_split_4012(n_days=5, stride=1, model='LSTM', diff=False)

    # make a prediction
    y_pred = model.predict(X_test)  # y_pred is the predicted return
    print(model.summary())

    # save result
    np.savetxt(f'../prediction/{surname}.csv', y_pred)

    # draw plot
    return_pred_plot(y_test, y_pred, model_num)
    price_pred_graph(y_pred, model_num)


# def plot_ROC_and_pred():
#
#     plt.figure(figsize=(10, 8))
#
#     y_test_class, y_pred_prob = CNN_classification(seed=int(seed.iloc[i]), cnn=int(cnn.iloc[i]), n_days=int(n_days.iloc[i]), stride=int(
#             stride.iloc[i]), model_type=model_type, diff=diff.iloc[i], epochs=int(epochs.iloc[i]), good='good')
#         fpr, tpr, thresholds = roc_curve(y_test_class, y_pred_prob)
#         plt.plot(fpr, tpr, label=model_list[i])
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend()
#     plt.show()

def evaluate_classification(seed,epochs,model_num):
    surname = f'lstm_model{model_num}_seed{seed}'
    modelname = '../model/' + surname + '.h5'
    model = load_model(modelname)
    X_train, X_test, y_train, y_test = train_test_split_4012(n_days=5, stride=1, model='LSTM', diff=False)

    # make a prediction
    y_pred = model.predict(X_test)

    y_pred_prob = 1 / (1 + np.exp(-y_pred))
    np.savetxt(f'../prediction/lstm_model{model_num}_seed{seed}_clssification2.csv', y_pred_prob)
    y_pred_class = np.where(y_pred_prob > 0.5, 1, 0)

    y_test_class = np.where(y_test > 0, 1, 0)

    print('Accuracy: ', accuracy_score(y_test_class, y_pred_class))
    print('Precision: ', precision_score(y_test_class, y_pred_class))
    print('Recall: ', recall_score(y_test_class, y_pred_class))
    print('F1: ', f1_score(y_test_class, y_pred_class))
    print('Confusion Matirx: ', confusion_matrix(y_test_class, y_pred_class))

    fpr, tpr, thresholds = roc_curve(y_test_class, y_pred_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    filename =  f'../graph/LSTM_graph/ROC_lstm_model{model_num}'
    plt.savefig(filename, dpi=1200, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # evaluate_regression(seed=619,epochs=100,model_num=7)
    evaluate_classification(seed=619,epochs=100,model_num=7)
