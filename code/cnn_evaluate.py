import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
from train_test_split import train_test_split_4012

model = load_model(
    '../model/cnn_model1seed4012_epochs100_days5_stride1_diffFalse.h5')
X_train, X_test, y_train, y_test = train_test_split_4012(
    n_days=5, stride=1, model='CNN', diff=False)

# make a prediction
y_pred = model.predict(X_test)
print(model.summary())

# plot prediction only display 5 x-axis label
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


y_pred_return = pd.Series(y_pred.flatten(), index=y_test.index)
def price_pred_graph(return_pred):

    close = pd.read_csv('../data/data.csv',index_col=0)['close']
    y_true = close.iloc[-276:-6]
    print(y_true)

    y_pred = []
    y_pred.append(y_true.iloc[0])

    i = 1
    while i<len(return_pred)-1:
        y_pred.append(y_pred[i-1]*(1+float(return_pred.iloc[i-1].values)))
        i+=1

    print(y_pred)
    plt.plot(y_true,label='close')
    plt.plot(y_pred,label='pred')
    plt.xticks(y_true.index[::int(len(y_true) / 5)])
    plt.show()
