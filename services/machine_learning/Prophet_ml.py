import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

stock_data = pd.read_json("Desktop/LINKUSDT_1d 2.json")
stock_data

stock_data["Date"] = pd.to_datetime(stock_data["open time"],unit='ms')
stock_data.index = stock_data["Date"]
# stock_data.index = pd.DatetimeIndex(stock_data.index).to_period('D')
stock_data

stock_data.drop(columns=['open time', 'close time', 'delete', 'Date'], inplace=True)
stock_data

from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

stock_data.head()

stock_data.drop(['open','DATE_FROM','high',
                    'low','volume','quote asset volume',
                    'number of trades','taker buy base asset volume','taker buy quote asset volume'], axis = 1, inplace = True, errors = 'ignore')
stock_data.head()

stock_data['ds'] = stock_data.index
stock_data.head()

stock_data['y'] =stock_data.close	

yy=stock_data[['ds','close']].tail(16)

yy

m = Prophet()


m.fit(stock_data)

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)


forecast[['ds' ,'yhat','yhat_lower', 'yhat_upper']].tail()

a = forecast[(forecast.ds >= '2020-01-16') & (forecast.ds < '2024-11-10')]

a[['ds', 'yhat','yhat_lower', 'yhat_upper']].tail(30)

m.plot(forecast)

m.plot_components(forecast)



from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Ensure the forecast dataframe has predictions for the test period
# Use only dates present in the dataset for comparison
test_data = stock_data[(stock_data['ds'] >= '2020-01-16') & (stock_data['ds'] < '2024-11-10')]

# Merge actuals with predictions
results = pd.merge(test_data[['ds', 'y']], forecast[['ds', 'yhat']], on='ds')

# Calculate error metrics
mae = mean_absolute_error(results['y'], results['yhat'])
mse = mean_squared_error(results['y'], results['yhat'])
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Optional: Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(results['ds'], results['y'], label='Actual')
plt.plot(results['ds'], results['yhat'], label='Predicted')
plt.legend()
plt.title('Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid()
plt.show()


# Select relevant forecast data
forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Define file path
output_path = "services/machine_learning/src/forecast_Prophet.csv"

# Save to CSV
forecast_output.to_csv(output_path, index=False)

print(f"Forecast results saved to {output_path}")


