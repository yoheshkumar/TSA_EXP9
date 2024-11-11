### DEVELOPED BY : YOHESH KUMAR R.M
### REGISTER NO : 212222240118
# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on MentalHealthSurvey using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of mental Health 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

file_path = 'Microsoft_Stock.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
series = data['Close']

result = adfuller(series)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

plt.figure(figsize=(12, 6))
plt.subplot(311)
plt.plot(series)
plt.title("Microsoft Stock Close Price Series")

plt.subplot(312)
plot_acf(series, ax=plt.gca(), lags=20)
plt.title("ACF Plot")

plt.subplot(313)
plot_pacf(series, ax=plt.gca(), lags=20)
plt.title("PACF Plot")
plt.tight_layout()
plt.show()

if result[1] > 0.05:
    series_diff = series.diff().dropna()
else:
    series_diff = series

model = ARIMA(series, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')

plt.figure(figsize=(12, 6))
plt.plot(series, label="Original Series")
plt.plot(forecast_index, forecast, label="Forecast", color='red')
plt.title("ARIMA Model Forecast on Microsoft Stock Close Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()


```
### OUTPUT:
![image](https://github.com/user-attachments/assets/da151401-648f-4162-92cd-9e6e96e72175)
![image](https://github.com/user-attachments/assets/f80e4739-c738-453b-8a00-03e83744923a)
![image](https://github.com/user-attachments/assets/8576e510-f4e6-4da8-a1e9-2c11c0c99279)
### RESULT:
Thus the program run successfully based on the ARIMA model using python.
