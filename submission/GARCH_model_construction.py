import pandas as pd
import numpy as np
import arch
import matplotlib.pyplot as plt
from datetime import datetime

data = pd.read_csv('data.csv').dropna(how='any')
scaled_returns = data['return'].values*100 # times 100 to scale
garch = arch.arch_model(scaled_returns, vol='garch', p=1, o=0, q=1)
garch_fitted = garch.fit(last_obs=1076) # fit model with data to 2022-03-08
garch_forecast = garch_fitted.forecast(horizon=5, reindex=False, start=1081)
predicted_mean = garch_forecast.mean['h.5']/100
predicted_sigma = (garch_forecast.residual_variance['h.5']**0.5)/100
predicted_return = (predicted_mean + predicted_sigma)

true = data['return'][1081:].values
predict = predicted_return.values
mse = sum((predict-true)**2)/len(true)

#plot
date = pd.to_datetime(data['date'].iloc[1081:])
plt.plot(date, true, label='true')
plt.plot(date, predict, label='predict')
plt.tick_params(axis='x', labelsize=8)
plt.xticks(rotation=-15)
plt.title('Time Series GARCH Model')
plt.legend()
plt.show()