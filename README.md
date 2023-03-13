# Stat4012 Group-Project

## Overview

>  To synthesize profitable trading strategies, accurately predicting future stock prices is crucial. There are mainly three approaches to make such predictions: 
>
> - cross-sectional regressions (OLS)
>   - cons: low accuracy due to the violation of OLS assumptions
> - time series models (e.g. ARIMA)
>   - cons: don't work well on volatile data
> - machine learning models. Among these
>   - machine learning models have shown higher accuracy even for volatile data
> - Our goal is to compare the prediction accuracy of two machine learning algorithms, Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM), and find the optimal model to conduct multivariate time series forecasting.

## Design

### Data

We will be using a single stock/ETF's historical pricing data (High, Low, Open, Close, Volume) for one year, at 5-minute intervals.

### Logistic

We will use High, Low, Close, and Volume data as regressors (features) and use Open data as the response.

### Hyperparameters

The hyperparameters we will be testing are:

1. Number of layers (hidden layer=1)
2. Number of neurons in each layer
3. Batch size
4. Activation function
5. Optimizer

### Evaluating Measures

We will be evaluating the accuracy and generalization error of the models.

### Questions to Answer

1. How to decide the number of layers and neurons in each layer?[^2]

## Implementation

### Steps

1. Collect data
2. Preprocess data by calculating returns and converting it into a stationary sequence
3. Initialize a neural network architecture through xxx
4. Prune unnecessary nodes

## References

1. [Analysis and Forecasting of Financial Time Series Using CNN and LSTM-Based Deep Learning.](https://link.springer.com/chapter/10.1007/978-981-16-4807-6_39)

2. [Integrating Fundamental and Technical Analysis of Stock Market through Multi-layer Perceptron.](https://ieeexplore.ieee.org/abstract/document/8488440)

3. [How to choose the number of hidden layers and nodes in a feedforward neural network?](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw) 

   [^1]: 1. the number of hidden layers equals one; and 2. the number of neurons in that layer is the mean of the neurons in the input and output layers.
   [^2]: Refer to Reference 3.
