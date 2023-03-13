# Stat4012 Group-Project

## Overview

>  To synthesize profitable trading strategies, accurately predicting future stock prices is crucial. There are mainly three approaches to make such predictions: 
   cross-sectional regressions, traditional time series models and machine learning models. 
   
>  Since machine learning models have shown higher accuracy even for volatile data, we hope to find an effective stock price forecasting model by comparing the prediction accuracy of two deep learning algorithms: Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM). We will also compare the performance of our selected model with the Random Forest algorithm to test its capability.
>  
## Design

- Data

  We will be using the historical pricing data of Tencent(0700.HK) for one year, at 5-minute intervals.

- Logistic

  We will set the length of the moving window to be 5 days, i.e. take the price data of previous 5 days as input and predict the price for the following day

- Hyperparameters 
 
  - Number of layers
  - Number of neurons in each layer
  - Activation function
  - Optimizer

- Evaluating Measures

  We will first evaluate models by comparing their accuracy and generalization error 
  Next, we will conduct hypothesis tests such as the Diebold and Mariano (1995) and Giacomini and White (2006) tests to examine the relative predictive ability of different models.


## Implementation

### Data Collection

Stock price and relative information will be download from Bloomberg terminal or [Yahoo Finance](https://finance.yahoo.com/). We will collect Tencent's historical pricing data (High, Low, Open, Close, Volume) at 5-minute intervals, in A-Share market for past one year. 

### Data Preprocessing

Data will be preprocessed by considering the missing data, outliers, stock split and dividend. Returns of the stock or ETF will be calculated and converted  into a stationary time series sequence. Then it will be divided into training set, and testing set with rate 8:2 for further analysis. 

### Neural Network Development

Initialize a neural network architecture through two chosen model CNN and LSTM. 

- For CNN model, # of layer, optimizer, hyperparameters. 

1. Prune unnecessary nodes

### Model Training

### Model Comparision

Models using algorithms CNN, LSTM and Random Forest will be tested on the testing set respectively. We will conduct hypothesis tests following Diebold and Mariano (1995) and Giacomini and White (2006) to examine the relative predictive ability of different models. Accuracy, Preceision, Recall, F1 score and ROC will be used to measure the performance of each model.



## References

1. [Analysis and Forecasting of Financial Time Series Using CNN and LSTM-Based Deep Learning.](https://link.springer.com/chapter/10.1007/978-981-16-4807-6_39)

2. [Integrating Fundamental and Technical Analysis of Stock Market through Multi-layer Perceptron.](https://ieeexplore.ieee.org/abstract/document/8488440)

3. [How to choose the number of hidden layers and nodes in a feedforward neural network?](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw) 

4. [Convolution on neural networks for high-frequency trend prediction of cryptocurrency exchange rates using technical indicators](https://www.sciencedirect.com/science/article/pii/S0957417420300750?via%3Dihub#bib0018)
   [^1]: the number of hidden layers equals one; and the number of neurons in that layer is the mean of the neurons in the input and output layers.
   [^2]: Refer to Reference 3.
