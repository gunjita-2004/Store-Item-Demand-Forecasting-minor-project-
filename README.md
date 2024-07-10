# Store-Item-Demand-Forecasting-minor-project-
import pandas as pd
import numpy as np
# Example: Load your data
data = {
    'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'sales': np.random.randint(1, 100, size=100)
}
df = pd.DataFrame(data)
df.set_index('date', inplace=True)

import matplotlib.pyplot as plt

df['sales'].plot(figsize=(12, 6))
plt.title('Sales Data')
plt.show()

from statsmodels.tsa.arima.model import np

# Define the model
model = np(df['sales'], order=(5, 1, 0))  # p, d, q values

# Fit the model
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Forecasting the next 10 days
forecast = model_fit.forecast(steps=10)

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['sales'], label='Historical Sales')
plt.plot(pd.date_range(start=df.index[-1], periods=11, freq='D')[1:], forecast, label='Forecasted Sales')
plt.legend()
plt.title('Sales Forecast')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import np

# Example: Load your data
data = {
    'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'sales': np.random.randint(1, 100, size=100)
}
df = pd.DataFrame(data)
df.set_index('date', inplace=True)

# Plot the sales data
df['sales'].plot(figsize=(12, 6))
plt.title('Sales Data')
plt.show()

# Fit an ARIMA model
model = np(df['sales'], order=(5, 1, 0))  # p, d, q values
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Forecasting the next 10 days
forecast = model_fit.forecast(steps=10)

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['sales'], label='Historical Sales')
plt.plot(pd.date_range(start=df.index[-1], periods=11, freq='D')[1:], forecast, label='Forecasted Sales')
plt.legend()
plt.title('Sales Forecast')
plt.show()
