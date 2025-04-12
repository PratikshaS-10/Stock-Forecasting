import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
dataset = pd.read_csv("AAPL_historical_data.csv")

# Convert 'Date' to datetime
dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce', utc=True)

# Check for missing and duplicate values
print(dataset.isnull().sum())
print(dataset.duplicated().sum())

# Closing price over time
sns.lineplot(x=dataset['Date'], y=dataset['Close'])
plt.ylabel('Closing Price')
plt.title('AAPL Stock Price from 1980-2025')
plt.show() 

# Check for stationarity
def check_stationary(timeseries):
    result = adfuller(timeseries)
    ADF_statistic = result[0]
    p_value = result[1]
    print(f'ADF Statistic: {ADF_statistic}')
    print(f'P-value: {p_value}')
    if p_value < 0.05:
        print("The time series is likely to be stationary.")
    else:
        print("The time series is non-stationary.")

check_stationary(dataset['Close'])

# Differencing to make the series stationary
dataset['diff'] = dataset['Close'].diff()
dataset.dropna(inplace=True)

check_stationary(dataset['diff'])


# Remove outliers from the original 'Close' prices
Q1 = dataset['diff'].quantile(0.25)
Q3 = dataset['diff'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = dataset[(dataset['diff'] >= lower_bound) & (dataset['diff'] <= upper_bound)]


best_aic = np.inf
best_order = None

# Grid search over p, d, q values
for d in range(0,2):
    for p in range(0,4):
        for q in range(0,4):
            try:
                # Fit ARIMA model
                model = ARIMA(stat_daily_sales, order=(p, d, q))
                model_fit = model.fit()
                # Get AIC
                aic = model_fit.aic
                # Update best AIC and parameters if current AIC is lower
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
            except:
                continue

print(f'Best ARIMA order: {best_order} with AIC: {best_aic}')

# Fit the ARIMA model
model = ARIMA(df_cleaned['Close'], order=(2, 1, 2))  # Use the cleaned data
model_fit = model.fit()
print(model_fit.summary())
fitted_values=model_fit.fittedvalues


forecast = model_fit.forecast(steps=3650)
print(forecast.head())  # Print the first few forecasted values

# Create a date range for the forecasted values
last_date = df_cleaned['Date'].iloc[-1]  # Get the last date from the cleaned data
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=3650, freq='B')  # Business days

# Create a DataFrame for the forecasted data
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Values': forecast})

# Plot the original data, fitted values, and forecast
# plt.figure(figsize=(12, 6))
plt.plot(df_cleaned['Date'], df_cleaned['Close'], label='Original Data', color='blue')
plt.plot(df_cleaned['Date'], model_fit.fittedvalues, label='Fitted Values', color='orange')
plt.title('AAPL Stock Price original v/s fitted values')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

#forecasted values
plt.plot(forecast_dates,forecast,label='Forecasted Values',color='green')
plt.title('AAPL Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# Residuals
residuals = model_fit.resid
sns.scatterplot(x=residuals.index, y=residuals)
plt.title('Residuals of the ARIMA Model')
plt.xlabel('Index')
plt.ylabel('Residuals')
plt.show()
check_stationary(residuals)


hw_model = ExponentialSmoothing(df_cleaned['Close'], trend='add', seasonal='add', seasonal_periods=252)
hw_fit = hw_model.fit()
fitted_values_exp = hw_fit.fittedvalues

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df_cleaned['Date'], df_cleaned['Close'], label='Original Data', color='blue')
plt.plot(df_cleaned['Date'], fitted_values_exp, label='Fitted Values', color='orange')
plt.title('AAPL Stock Price: Original vs Fitted vs Forecasted Values')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

#forecasted values
hw_forecast = hw_fit.forecast(steps=3650)

# Dates for forecast
last_date = df_cleaned['Date'].iloc[-1]  # Get the last date from the cleaned data
forecast_dates_hw = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=3650, freq='B')
plt.plot(forecast_dates_hw,hw_forecast,label='Forecasted Values',color='green')
plt.title('AAPL Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

residuals2 = hw_fit.resid
sns.scatterplot(x=residuals2.index, y=residuals2)
plt.title('Residuals of the Exponentiall Smoothing')
plt.xlabel('Index')
plt.ylabel('Residuals')
plt.show()
check_stationary(residuals)
#arims v*/s exp--check
plt.plot(forecast_dates, forecast, label='ARIMA Forecast', color='green')
plt.plot(forecast_dates_hw, hw_forecast, label='Holt-Winters Forecast', color='purple')
plt.title('ARIMA vs Holt-Winters Forecast (Next 14 Years)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show() 


mse_arima = mean_squared_error(df_cleaned['Close'], fitted_values)
print(f'Mean Squared Error for ARIMA: {mse_arima:.4f}')

mse_hw = mean_squared_error(df_cleaned['Close'], fitted_values_exp)
print(f'Mean Squared Error for Holt-Winters: {mse_hw:.4f}')

#Prophet
prophet_data = df_cleaned[['Date', 'Close']].copy()
prophet_data['Date'] = prophet_data['Date'].dt.tz_localize(None)
prophet_data = prophet_data.rename(columns={'Date': 'ds', 'Close': 'y'})

prophet_model = Prophet(daily_seasonality=False, yearly_seasonality=True)
prophet_model.fit(prophet_data)

# Forecast 3650 business days
future = prophet_model.make_future_dataframe(periods=3650)
forecast = prophet_model.predict(future)
last_date = prophet_data['ds'].max().replace(tzinfo=None)

# Filter forecasted values that are after the last date in the original dataset
future_forecast = forecast[forecast['ds'] > last_date]


fitted = forecast['yhat'].iloc[:len(prophet_data)]

plt.figure(figsize=(12, 6))
plt.plot(prophet_data['ds'], prophet_data['y'], label='Actual', color='black')
plt.plot(prophet_data['ds'], fitted, label='Fitted (Prophet)', color='blue')
plt.plot(future_forecast['ds'], future_forecast['yhat'], label='Forecast (Prophet)', color='red')
plt.title('Prophet Forecast with Actuals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# MSE on fitted values

mse_prophet = mean_squared_error(df_cleaned['Close'], fitted)
print(f'MSE for Prophet: {mse_prophet:.4f}')

data = dataset[['Date', 'Close']].dropna()

# Sort data by date
data = data.sort_values(by='Date')

# Use only 'Close' price for forecasting
prices = data[['Close']].values

# Normalize the data (scale to range [0,1])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create sequences of the stock price (LSTM expects sequences of data)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])  # Create sequence
        y.append(data[i, 0])  # Actual price
    return np.array(X), np.array(y)

# Set sequence length (e.g., 60 days to predict next day's price)
sequence_length = 60
X, y = create_sequences(scaled_prices, sequence_length)

# Reshape X to be compatible with LSTM (samples, timesteps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))  # Dropout to avoid overfitting
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (fit to data)
history = model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Predict stock prices on the training set
predicted_prices = model.predict(X)

# Inverse the scaling to get original values
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

# Plot the actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(data['Date'].iloc[sequence_length:], actual_prices, label='Actual Price', color='black')
plt.plot(data['Date'].iloc[sequence_length:], predicted_prices, label='LSTM Predicted Price', color='blue')
plt.title('LSTM: Actual vs Predicted Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Forecasting the next 30 days
# Use the last `sequence_length` data points for prediction
last_data = scaled_prices[-sequence_length:]

# Predict the next 'n' days (e.g., next 30 days)
n_days = 30
predictions = []

for _ in range(n_days):
    # Reshape for prediction
    last_data_reshaped = np.reshape(last_data, (1, sequence_length, 1))

    # Predict next price
    next_price = model.predict(last_data_reshaped)

    # Inverse transform to get actual price
    next_price = scaler.inverse_transform(next_price)
    predictions.append(next_price[0][0])

    # Update `last_data` with the predicted price
    last_data = np.append(last_data[1:], next_price)

# Create a date range for the forecasted data
forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_days)

# Plot the forecasted values
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Actual Price', color='black')
plt.plot(forecast_dates, predictions, label='LSTM Forecasted Price', color='blue')
plt.title(f'LSTM Forecast for {n_days} Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

