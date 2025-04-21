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

dataset = pd.read_csv("AAPL_historical_data.csv")
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

# Remove outliers 
Q1 = dataset['diff'].quantile(0.25)
Q3 = dataset['diff'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = dataset[(dataset['diff'] >= lower_bound) & (dataset['diff'] <= upper_bound)]
print(dataset.head())


best_aic = np.inf
best_order = None

# Grid search over p, d, q values
for d in range(0,2):
    for p in range(0,3):
        for q in range(0,3):
            try:
                # Fit ARIMA model
                model = ARIMA(df_cleaned['diff'], order=(p, d, q))
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

#Fit the ARIMA model
model = ARIMA(df_cleaned['Close'], order=(1, 1, 2))  # Use the cleaned data
model_fit = model.fit()
print(model_fit.summary())
fitted_values=model_fit.fittedvalues


forecast1 = model_fit.forecast(steps=252)
print(forecast1.head())  # Print the first few forecasted values

# Create a date range for the forecasted values
last_date = df_cleaned['Date'].iloc[-1]  # Get the last date from the cleaned data
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=252, freq='B')  # Business days

# Create a DataFrame for the forecasted data
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Values': forecast1})

# Plot the original data, fitted values, and forecast
# plt.figure(figsize=(12, 6))
plt.plot(df_cleaned['Date'], df_cleaned['Close'], label='Original Data', color='blue')
plt.plot(df_cleaned['Date'], model_fit.fittedvalues, label='Fitted Values', color='orange')
plt.title('AAPL Stock Price original v/s fitted values-ARIMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

#forecasted values
plt.plot(forecast_dates,forecast1,label='Forecasted Values',color='green')
plt.title('AAPL Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# Residuals
residuals = model_fit.resid
print(residuals)
sns.scatterplot(x=df_cleaned['Date'], y=residuals)
plt.title('Residuals of the ARIMA Model')
plt.xlabel('Dates')
plt.ylabel('Residuals')
plt.show()
check_stationary(residuals)

#EXPONENTIAL SMOOTHING
hw_model = ExponentialSmoothing(df_cleaned['Close'], trend='add', seasonal='add', seasonal_periods=252)
hw_fit = hw_model.fit()
fitted_values_exp = hw_fit.fittedvalues

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df_cleaned['Date'], df_cleaned['Close'], label='Original Data', color='blue')
plt.plot(df_cleaned['Date'], fitted_values_exp, label='Fitted Values', color='orange')
plt.title('Original vs Fitted-Exponential Smoothing ')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

#forecasted values
hw_forecast = hw_fit.forecast(steps=252)

# Dates for forecast
last_date = df_cleaned['Date'].iloc[-1]  # Get the last date from the cleaned data
forecast_dates_hw = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=252, freq='B')
plt.plot(forecast_dates_hw,hw_forecast,label='Forecasted Values',color='green')
plt.title('Forecast-EXPONENTIAL SMOOTHING ')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

residuals2 = hw_fit.resid
sns.scatterplot(x=df_cleaned['Date'],y=residuals2)
plt.title('Residuals of the Exponential Smoothing')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.show()
check_stationary(residuals)

#Prophet
prophet_data = df_cleaned[['Date', 'Close']].copy()
prophet_data['Date'] = prophet_data['Date'].dt.tz_localize(None)

prophet_data = prophet_data.rename(columns={'Date': 'ds', 'Close': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_data)


# Forecast 252 business days
future = prophet_model.make_future_dataframe(periods=252)
forecast2 = prophet_model.predict(future)
last_date = prophet_data['ds'].max()

future_forecast = forecast2[forecast2['ds'] > last_date]
fitted = forecast2['yhat'].iloc[:len(prophet_data)]
plt.figure(figsize=(12, 6))
plt.plot(prophet_data['ds'], prophet_data['y'], label='Actual', color='black')
plt.plot(prophet_data['ds'], fitted, label='Fitted (Prophet)', color='blue')
plt.title('Prophet Forecast with Actuals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.plot(future_forecast['ds'], future_forecast['yhat'], label='Forecast (Prophet)', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


#resuiduals
prophet_data['residuals'] = prophet_data['y'] - fitted.values
plt.figure(figsize=(12, 4))
plt.scatter(prophet_data['ds'], prophet_data['residuals'], color='purple', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals Scatter Plot (Prophet)')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.grid(True)
plt.show()

#LSTM
data = dataset[['Date', 'Close']].dropna()
data = data.sort_values(by='Date')
# Use only 'Close' price for forecasting
prices = data[['Close']].values

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

# Forecasting the next 10 YEARS
# Use the last `sequence_length` data points for prediction
last_data = scaled_prices[-sequence_length:]

# Predict the next 'n' days (e.g., next 252 days)
n_days = 252
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
#plt.plot(data['Date'], data['Close'], label='Actual Price', color='black')
plt.plot(forecast_dates, predictions, label='LSTM Forecasted Price', color='blue')
plt.title(f'LSTM Forecast for {n_days} Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

residuals4 = actual_prices.flatten() - predicted_prices.flatten()
plt.figure(figsize=(12, 4))
sns.scatterplot(x=data['Date'].iloc[sequence_length:], y=residuals4, color='purple', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('LSTM Residuals (Actual - Predicted)')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.grid(True)
plt.show()

mse_arima = mean_squared_error(df_cleaned['Close'], model_fit.fittedvalues)
mse_hw = mean_squared_error(df_cleaned['Close'], hw_fit.fittedvalues)
mse_prophet = mean_squared_error(df_cleaned['Close'], fitted)
mse_lstm = mean_squared_error(data['Close'].iloc[sequence_length:], predicted_prices.flatten())

print(f"ARIMA MSE: {mse_arima:.2f}")
print(f"Holt-Winters MSE: {mse_hw:.2f}")
print(f"Prophet MSE: {mse_prophet:.2f}")
print(f"LSTM MSE: {mse_lstm:.2f}")

