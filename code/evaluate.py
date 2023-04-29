import pandas as pd
import matplotlib.pyplot as plt

return_pred = pd.read_csv('../data/y_pred_return.csv',index_col=0)
print(return_pred)

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
