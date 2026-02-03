import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

forecast_size = 744

data = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/econ8310-assignment1/refs/heads/main/assignment_data_train.csv")
format = '%Y-%m-%d %H:%M:%S'
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format=format)
data.set_index(pd.DatetimeIndex(data['Timestamp']), inplace=True)

model = ExponentialSmoothing(data['trips'], trend='mul', damped_trend=True, seasonal='add', seasonal_periods=24*7, freq='h')
modelFit = model.fit()
pred = modelFit.forecast(forecast_size)

