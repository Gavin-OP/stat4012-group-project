import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from train_test_split import train_test_split_4012
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os


def predict_price(seed=4012, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False, epochs=100, model_num=0, good='NO'):
    if model_type == 'CNN':
        if good == 'NO':
            modelname = '../model/new_PCA_model/cnn_model' + str(cnn) + '_seed' + str(seed) + '_epochs' + str(epochs) +\
                '_days' + str(n_days) + '_stride' + str(stride) + \
                '_diff' + str(diff) + '.h5'
        elif good == 'good':
            modelname = '../model/new_PCA_model/cnn_model' + str(cnn) + '_seed' + str(seed) + '_epochs' + str(epochs) +\
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
        np.savetxt(f'../prediction/{surname}.csv', y_pred)
    elif model_type == 'CNN':
        if good == 'good':
            np.savetxt(
                f'../prediction/cnn_model{cnn}_seed{seed}_epochs{epochs}_days{n_days}_stride{stride}_diff{diff}_good.csv', y_pred)

    # return_pred_plot(y_test, y_pred, seed=seed, cnn=cnn, n_days=n_days, stride=stride, model_type=model_type, diff=diff, epochs=epochs)
    # price_pred_graph(y_pred, seed=seed, cnn=cnn, n_days=n_days,
    #                  stride=stride, model_type=model_type, diff=diff, epochs=epochs)
    return y_test, y_pred

# plot prediction only display 5 x-axis label


def return_pred_plot(y_test, y_pred, seed=4012, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False, epochs=100):
    y_test = pd.DataFrame(y_test)
    y_pred = pd.DataFrame(y_pred)
    y_test.index = range(len(y_test))
    y_pred.index = range(len(y_test))
    # plt.figure(figsize=(16, 9))
    # plt.plot(y_test.index, y_test, label='y_test')
    # plt.plot(y_test.index, y_pred, label='y_pred')
    # plt.xticks(y_test.index[::int(len(y_test) / 5)])
    # plt.legend()
    # filename = '../graph/better_model_cnn_ROC/return_cnn_model' + str(cnn) + '_return_pred_seed' + str(seed) + '_epochs' + str(epochs) +\
    #     '_days' + str(n_days) + '_stride' + str(stride) + \
    #     '_diff' + str(diff) + '.png'
    # plt.savefig(filename, dpi=1200, bbox_inches='tight')
    # plt.show()


def price_pred_graph(return_pred, seed=4012, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False, epochs=100, good='NO'):
    return_pred = pd.DataFrame(return_pred)
    close = pd.read_csv('../data/data.csv', index_col=0)['close']
    y_true = close.iloc[-276:-6]

    y_pred = []
    y_pred.append(y_true.iloc[0])

    i = 1
    while i < len(return_pred)-1:
        y_pred.append(y_pred[i-1]*(1+float(return_pred.iloc[i-1].values)))
        i += 1

    # plt.figure(figsize=(16, 9))
    # plt.plot(y_true, label='close')

    label = 'cnn_model' + str(cnn) + '_seed' + str(seed) + '_epochs' + str(epochs) +\
        '_days' + str(n_days) + '_stride' + str(stride) + \
        '_diff' + str(diff)
    plt.plot(y_pred, label=label)

    # plt.plot(y_pred, label='prediction')
    # plt.xticks(y_true.index[::int(len(y_true) / 5)])
    # plt.legend()
    # filename = '../graph/better_model_cnn_ROC/price_cnn_model' + str(cnn) + '_price_pred_seed' + str(seed) + '_epochs' + str(epochs) +\
    #     '_days' + str(n_days) + '_stride' + str(stride) + \
    #     '_diff' + str(diff) + '.png'
    # plt.savefig(filename, dpi=1200, bbox_inches='tight')
    # plt.show()


def CNN_classification(seed=4012, cnn=1, n_days=5, stride=1, model_type='CNN', diff=False, epochs=100, model_num=0, good='NO'):
    if model_type == 'CNN':
        y_test, y_pred = predict_price(seed=seed, cnn=cnn, n_days=n_days, stride=stride,
                                   model_type=model_type, diff=diff, epochs=epochs, good=good)
    elif model_type == 'LSTM':
        y_test, y_pred = predict_price(seed=seed,cnn=1, n_days=5, stride=stride, model_type=model_type, diff=False, epochs=epochs, model_num=model_num, good='NO')

    y_pred_prob = 1 / (1 + np.exp(-y_pred))
    y_pred_class = np.where(y_pred_prob > 0.5, 1, 0)
    y_test_class = np.where(y_test > 0, 1, 0)

    print('Accuracy: ', accuracy_score(y_test_class, y_pred_class))
    print('Precision: ', precision_score(y_test_class, y_pred_class))
    print('Recall: ', recall_score(y_test_class, y_pred_class))
    print('F1: ', f1_score(y_test_class, y_pred_class))
    print('Confusion Matirx: ', confusion_matrix(y_test_class, y_pred_class))

    # save accuracy, precision, recall, f1, confusion matrix in csv
    df = pd.DataFrame([accuracy_score(y_test_class, y_pred_class), precision_score(y_test_class, y_pred_class),
                       recall_score(y_test_class, y_pred_class), f1_score(y_test_class, y_pred_class)])
    df.to_csv('../data/new_PCA_result/cnn_model' + str(cnn) + '_seed' + str(seed) + '_epochs' + str(epochs) +
              '_days' + str(n_days) + '_stride' + str(stride) +
              '_diff' + str(diff) + '.csv')
    

    # plot roc curve
    # fpr, tpr, thresholds = roc_curve(y_test_class, y_pred_prob)
    # plt.figure(figsize=(16, 9))
    # plt.plot(fpr, tpr)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # filename = '../graph/better_model_cnn_ROC/ROC_cnn_model' + str(cnn) + '_ROC_seed' + str(seed) + '_epochs' + str(epochs) +\
    #     '_days' + str(n_days) + '_stride' + str(stride) + \
    #     '_diff' + str(diff) + '.png'
    # plt.savefig(filename, dpi=1200, bbox_inches='tight')
    # plt.show()
    return y_test_class, y_pred_prob


def plot_ROC_and_pred():
    # plot all models from model_list in one plot
    model_list = os.listdir('../model/new_PCA_model/good_new_PCA_model/better')
    model_list = [i for i in model_list if i[-3:] == '.h5']
    # extract seed, epochs, cnn, n_days, stride, model_type, diff from model name
    model_info = []
    for i in model_list:
        model_info.append(i.split('_'))
    model_info = pd.DataFrame(model_info)
    cnn = model_info[1].str.extract('(\d+)')
    n_days = model_info[4].str.extract('(\d+)')
    stride = model_info[5].str.extract('(\d+)')
    # diff will not include first 4 characters 'diff'
    diff = model_info[6].str[4:]
    epochs = model_info[3].str.extract('(\d+)')
    seed = model_info[2].str.extract('(\d+)')
    model_type = 'CNN'
    # plot all ROC curve in one plot
    # plt.figure(figsize=(16, 9))
    # for i in range(len(model_list)):
    #     # print model info
    #     print('model: ', model_list[i])
    #     y_test_class, y_pred_prob = CNN_classification(seed=int(seed.iloc[i]), cnn=int(cnn.iloc[i]), n_days=int(n_days.iloc[i]), stride=int(
    #         stride.iloc[i]), model_type=model_type, diff=diff.iloc[i], epochs=int(epochs.iloc[i]), good='good')
    #     fpr, tpr, thresholds = roc_curve(y_test_class, y_pred_prob)
    #     plt.plot(fpr, tpr, label=model_list[i])
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend()
    # save = '../graph/report/ROC_all_model.png'
    # plt.savefig(save, dpi=1200, bbox_inches='tight')
    # plt.show()

    # # plot all predicted return in one plot
    # plt.figure(figsize=(16, 9))
    # for i in range(len(model_list)):
    #     y_test, y_pred = predict_price(seed=int(seed.iloc[i]), cnn=int(cnn.iloc[i]), n_days=int(n_days.iloc[i]), stride=int(
    #         stride.iloc[i]), model_type=model_type, diff=diff.iloc[i], epochs=int(epochs.iloc[i]), good='good')
    #     # make y_test to pd.Series and give index as close price index
    #     close = pd.read_csv('../data/data.csv', index_col=0)['close']
    #     y_true = close.iloc[-276:-6]
    #     y_test = pd.Series(y_test, index=y_true.index)
    #     if i == 0:
    #         plt.plot(y_test, label='true')
    #     plt.plot(y_pred, label=model_list[i])
    # plt.xticks(y_test.index[::int(len(y_test) / 5)])
    # plt.legend()
    # save = '../graph/report/return.png'
    # plt.savefig(save, dpi=1200, bbox_inches='tight')
    # plt.show()

    # plot all predicted price in one plot
    # plt.figure(figsize=(16, 9))
    # for i in range(len(model_list)):
    #     y_test, y_pred = predict_price(seed=int(seed.iloc[i]), cnn=int(cnn.iloc[i]), n_days=int(n_days.iloc[i]), stride=int(
    #         stride.iloc[i]), model_type=model_type, diff=diff.iloc[i], epochs=int(epochs.iloc[i]), good='good')
    #     if i == 0:
    #         close = pd.read_csv('../data/data.csv', index_col=0)['close']
    #         y_true = close.iloc[-276:-6]
    #         plt.plot(y_true, label='true')

    #     price_pred_graph(y_pred, seed=int(seed.iloc[i]), cnn=int(cnn.iloc[i]), n_days=int(n_days.iloc[i]), stride=int(
    #         stride.iloc[i]), model_type=model_type, diff=diff.iloc[i], epochs=int(epochs.iloc[i]), good='good')
    # plt.xticks(y_true.index[::int(len(y_test) / 10)])
    # # rotate xticks
    # plt.xticks(rotation=45)
    # plt.legend()
    # save = '../graph/report/price.png'
    # plt.savefig(save, dpi=1200, bbox_inches='tight')
    # plt.show()


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
    # y_test, y_pred = predict_price(seed=998, epochs=100, cnn=2, n_days=5, stride=1, model_type='CNN', diff=False, good='good')
    # y_test, y_pred = predict_price(seed=61, epochs=100, cnn=4, n_days=5, stride=1, model_type='CNN', diff=True, good='good')
    # plot_ROC_and_pred()
    model_list = os.listdir('../model/new_PCA_model/good_new_PCA_model/better')
    model_list = [i for i in model_list if i[-3:] == '.h5']
    # extract seed, epochs, cnn, n_days, stride, model_type, diff from model name
    model_info = []
    for i in model_list:
        model_info.append(i.split('_'))
    model_info = pd.DataFrame(model_info)
    cnn = model_info[1].str.extract('(\d+)')
    n_days = model_info[4].str.extract('(\d+)')
    stride = model_info[5].str.extract('(\d+)')
    # diff will not include first 4 characters 'diff'
    diff = model_info[6].str[4:]
    epochs = model_info[3].str.extract('(\d+)')
    seed = model_info[2].str.extract('(\d+)')
    model_type = 'CNN'

    for i in range(len(model_list)):
        # print model info
        print('model name: ', model_list[i])
        CNN_classification(seed=int(seed.iloc[i]), cnn=int(cnn.iloc[i]), n_days=int(n_days.iloc[i]), stride=int(
            stride.iloc[i]), model_type=model_type, diff=diff.iloc[i], epochs=int(epochs.iloc[i]), good='good')



    # LSTM test
    # y_test, y_pred = predict_price(seed=619,epochs=100,model_type='LSTM',model_num=7)
    # return_pred_plot(y_test, y_pred)
    # price_pred_graph(y_pred)

    # CNN_classification(seed=619,epochs=100,model_type='LSTM',model_num=7)
    # CNN_classification(seed=619, cnn=1, n_days=5, stride=1, model_type='LSTM', diff=False, epochs=100, model_num=7, good='NO')