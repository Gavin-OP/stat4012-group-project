# Stat4012 Group-Project

## Overview

>  To synthesize profitable trading strategies, accurately predicting future stock prices is crucial. There are mainly three approaches to make such predictions: 
>
> - cross-sectional regressions (OLS)
>    - cons: low accuracy due to the violation of OLS assumptions
> - time series models (e.g. ARIMA)
>    - cons: don't work well on volatile data
> - machine learning models.
>    - machine learning models have shown higher accuracy even for volatile data
> - Our goal is to compare the prediction accuracy of two machine learning algorithms, Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM), and find the optimal model to conduct multivariate time series forecasting mainly by Python. 

## Design

- Data

  We will be using a single stock/ETF's historical pricing data (High, Low, Open, Close, Volume) for one year, at 5-minute intervals.

- Logistic

  We will use High, Low, Close, and Volume data as regressors (features) and use Open data as the response.

- Hyperparameters

  The hyperparameters we will be testing are:

  - Number of layers
  - Number of neurons in each layer
  - Batch size
  - Activation function
  - Optimizer

- Evaluating Measures

  We will be evaluating the accuracy and generalization error of the models.

- Questions to Answer
  - How to decide the number of layers and neurons in each layer?[^2]

## Implementation

### Data Collection

Stock price and relative information will be download from Bloomberg terminal or [Yahoo Finance](https://finance.yahoo.com/). We will collect a single stock or ETF's historical pricing data (High, Low, Open, Close) at 5-minute intervals, in A-Share market for past one year. 

### Data Preprocessing

Data will be preprocessed by considering the missing data, outliers, stock split and dividend. Returns of the stock or ETF will be calculated and converted  into a stationary time series sequence. Then it will be divided into training set, and testing set with rate 80:20 for further analysis. 

### Neural Network Development

Initialize a neural network architecture through two chosen model CNN and LSTM. 

- For CNN model, # of layer, optimizer, hyperparameters. 

1. Prune unnecessary nodes

### Model Training



## References

1. [Analysis and Forecasting of Financial Time Series Using CNN and LSTM-Based Deep Learning.](https://link.springer.com/chapter/10.1007/978-981-16-4807-6_39)

2. [Integrating Fundamental and Technical Analysis of Stock Market through Multi-layer Perceptron.](https://ieeexplore.ieee.org/abstract/document/8488440)

3. [How to choose the number of hidden layers and nodes in a feedforward neural network?](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw) 

   [^1]: the number of hidden layers equals one; and the number of neurons in that layer is the mean of the neurons in the input and output layers.
   [^2]: Refer to Reference 3.
