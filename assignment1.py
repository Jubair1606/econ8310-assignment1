import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -----------------------------
# Load training data
# -----------------------------
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
train = pd.read_csv(train_url)

# Use trips as the time series
y = train["trips"].astype(float)

# -----------------------------
# Define the model
# -----------------------------
model = ExponentialSmoothing(
    y,
    trend="add",
    seasonal="mul",
    seasonal_periods=24
)

# -----------------------------
# Fit the model
# -----------------------------
modelFit = model.fit(optimized=True)

# -----------------------------
# Forecast 744 hours (January)
# -----------------------------
pred = modelFit.forecast(744)

# Ensure correct format
pred = np.array(pred, dtype=float)
