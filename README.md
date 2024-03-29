# Stat4012 Group-Project

## Overview

>  To synthesize profitable trading strategies, accurately predicting future stock prices is crucial. There are mainly three approaches to make such predictions: cross-sectional regressions, traditional time series models and machine learning models. 

   Since machine learning models have shown higher accuracy even for volatile data, we hope to find an effective stock price forecasting model by comparing the prediction accuracy of two deep learning algorithms: Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM). We will also compare the performance of our selected model with the Random Forest algorithm to test its capability.

## Project Design

### Data Collection

Stock price and relative information will be download from Bloomberg terminal or [Yahoo Finance](https://finance.yahoo.com/). A single stock (i.e. Tencent 0700.HK) historical pricing data (`High`, `Low`, `Open`, `Close`) and trading `Volume` at 5-minute intervals in Hong Kong stock market for past 3 year will be collected. `High`, `Low`,`Close` and ` Volume` will be the regressors (features) while `Open` price will be the response.  

Main Features: 

1. fundamental: O,H,L,C,V,换手率

2. technical:

   1. 趋势指标：指数移动平均（EMA), 平滑异同移动平均线（MACD), ROC, S_Roc

   2. 强弱指标：相对强弱指标（RSI), 劲道指数

3. 重要转折点：黄金交叉等 

4. 事件：股票增发、股东增持减持、派息等

   3&4 发生->1 未发生->0

### Data Preprocessing

Data will be preprocessed by considering the missing data, outliers, stock split and dividend. Returns of the stock will be calculated and converted  into a stationary time series sequence. Then it will be divided into training set and testing set with rate 8:2 for further analysis. 

Feature Engineering

1. standardization: ?最大最小标准化 or Z-score标准化
2. what else could we do?

### Neural Network Development

Our neural network model will start from univariate CNN with the return be the feature. Multivariate CNN model will be developed by combining features such as `Volume` after change the data shape. We also want to build a univariate LSTM model based on with previous returns as input to predict the `Open` price for the next few days. 

For each model, we will try different hyperparameters to find the model with the best performance, including number of layers, number of neurons in each layer, batch size, activation function, optimizer. Unnecessary nodes will also be pruned to improve the models. 

### Model Training

Train the models by backpropagation to minimize the loss function. 

### Model Comparison

Models using algorithms CNN, LSTM and Random Forest will be tested on the testing set respectively. We will conduct hypothesis tests following Diebold and Mariano (1995) and Giacomini and White (2006) to examine the relative predictive ability of different models. Accuracy, precision, recall, F1 score, MSE and ROC will be used to measure the performance of each model.

## To do

- Feature Engineering: Set different seeds to pick the accuracy.
- How to decide the number of layers and neurons in each layer?[^2]
- 预测price而非return 
- 2Dconvolutional
- 用return重新再生成一个stock price
- train-test-split
- Regularization
- 如果normalize之后，predict的时候应该如何处理？
- PCA之后的Comp1，Comp2应该和normalized之后的data结合还是之前结合

## References

1. [Analysis and Forecasting of Financial Time Series Using CNN and LSTM-Based Deep Learning.](https://link.springer.com/chapter/10.1007/978-981-16-4807-6_39)
2. [Integrating Fundamental and Technical Analysis of Stock Market through Multi-layer Perceptron.](https://ieeexplore.ieee.org/abstract/document/8488440)
3. [How to choose the number of hidden layers and nodes in a feedforward neural network?](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw) 
4. [Convolution on neural networks for high-frequency trend prediction of cryptocurrency exchange rates using technical indicators](https://www.sciencedirect.com/science/article/pii/S0957417420300750?via%3Dihub#bib0018)  
5. [CNNPred: CNN-based stock market prediction using several data sources](https://www.arxiv-vanity.com/papers/1810.08923/)  
6. [Stock Market Prediction using CNN and LSTM](https://cs230.stanford.edu/projects_winter_2021/reports/70667451.pdf)
7. [How to Develop Convolutional Neural Network Models for Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/)

   [^1]: the number of hidden layers equals one; and the number of neurons in that layer is the mean of the neurons in the input and output layers.
   [^2]: Refer to Reference 3.
