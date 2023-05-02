import pandas as pd
import numpy as np
import backtrader as bt
import datetime


class IntradayStrategy(bt.Strategy):
    params = (("printlog", True), )

    def log(self, txt, dt=None, doprint=False):
        """Logging function for strategy"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")

    def __init__(self):
        self.dataopen = self.datas[0].open
        self.dataclose = self.datas[0].close
        # The name 'volumn' is to satisfy the input requirements of Backtrader, what is included in this column is the sixth day return
        self.datavolume = self.datas[0].volume
        self.order = None
        self.open = 1

    def next(self):

        cash = self.broker.get_cash()
        size = int(cash * 0.6 / self.dataopen[0])

        if self.datavolume[0] > 0.03:
            self.order = 1
        else:
            self.order = 0

        if self.order == 0:
            return
        else:
            if self.open == 1:
                self.log('Open, %.2f' % self.dataopen[0])
                self.buy(size=size, exectype=bt.Order.Market)
                self.open = 0
            else:
                self.log('Close, %2f' % self.dataclose[0])
                self.close()
                self.open = 1


if __name__ == "__main__":

    # Read data
    originaldata = pd.read_csv('/Users/zhangpuming/Desktop/stat4012 project/backtrader/data.csv', index_col=0)
    originaldata.sort_values(by='date', ascending=True, inplace=True)
    newdata = originaldata.copy()
    newdata = newdata.reset_index()
    predicted_return = pd.read_csv('/Users/zhangpuming/Desktop/stat4012 project/backtrader/model/garch.csv', header=None)
    y_pred = predicted_return.values
    pred = []
    for i in range(len(y_pred)):
        pred.append(y_pred[i][0])
    newdata['pred'] = np.hstack((np.zeros(newdata[newdata.date=='2022-03-17'].index[0]), pred))
    array1 = np.repeat(newdata.values, 2, axis=0)
    for i in array1:
        i[0] = datetime.datetime.strptime(i[0], '%Y-%m-%d')
    for j in range(len(newdata.index)):
        array1[2 * j + 1][0] += datetime.timedelta(hours=14)
    newdata = pd.DataFrame(array1)
    temp = originaldata.reset_index()
    temp['volume'] = 0
    newdata.columns = temp.columns
    newdata = newdata[['date', 'open', 'close', 'volume']]
    newdata = newdata.set_index('date')

    # Add our strategy
    cerebro = bt.Cerebro()
    cerebro.addstrategy(IntradayStrategy)

    # Add data
    data1 = bt.feeds.PandasData(dataname=newdata['2022-03-17':'2023-04-13'])
    cerebro.adddata(data1)

    # Set commission fee rate to be 0.1% and principal to be $100,000
    cerebro.broker.setcommission(commission=0.001)
    cerebro.broker.setcash(100000.0)

    # Run the strategy
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='AnnualReturn')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.03, annualize=True, _name='SharpeRatio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DrawDown')
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    result = cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    strat = result[0]

    # Analysis
    print("--------------- AnnualReturn -----------------")
    print(strat.analyzers.AnnualReturn.get_analysis())
    print("--------------- SharpeRatio -----------------")
    print(strat.analyzers.SharpeRatio.get_analysis())
    print("--------------- DrawDown -----------------")
    print(strat.analyzers.DrawDown.get_analysis())
    cerebro.plot()
