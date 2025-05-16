# 📈 AAPL Stock Price Forecasting (1980–2025)

This project performs in-depth **time series forecasting** using **ARIMA**, **Exponential Smoothing** and **Facebook Prophet**, on **AAPL (Apple Inc.)** historical stock data. The goal is to explore trends, check stationarity, model the data, and forecast future stock prices for next trade year.

---

## 🔍 Features

- ✔️ Exploratory Data Analysis (EDA) and Visualization  
- 📊 Stationarity Testing using ADF (Augmented Dickey-Fuller)  
- 🔁 ARIMA Modeling with Grid Search  
- 🔮 Forecasting with:
  - ARIMA
  - Holt-Winters Exponential Smoothing
  - Prophet (Facebook)
- 🧠 Residual Analysis for each model
- 📉 Model Evaluation via Mean Squared Error (MSE)

---

## 📂 Dataset

- `AAPL_historical_data.csv`

---

## ⚙️ How to Run

1. Clone the repository or download the code.
2. Ensure the following libraries are installed:

```bash
pip install pandas numpy seaborn matplotlib statsmodels prophet scikit-learn tensorflow
```

3. Run the script:

```bash
python forecast_aapl_stock.py
```

---

## 🧪 Models & Methods

### ✅ ARIMA
- Uses grid search to determine best `(p,d,q)` values.
- Provides fitted values and residuals.
- Forecasts next 252 business days.

### ✅ Holt-Winters (Exponential Smoothing)
- Handles trend and seasonality.
- Forecasts next 252 business days.

### ✅ Facebook Prophet
- Automatically detects trends, holidays, and seasonality.
- Easily visualizes component trends and residuals.

---

## 📈 Evaluation

Mean Squared Error (MSE) and Root Mean Squared Error(RMSE) is calculated for each model to assess accuracy.

```text
                MSE        RMSE
ARIMA :        0.68        0.82
Holt-Winters : 0.66        0.81
Prophet :      33.50       5.79

```

---

## 📊 Visualizations

- AAPL stock price over time
- Residual plots for each model
- Actual vs Fitted plots
- Forecast trends per model

---
Built as a comprehensive exploration of time series forecasting models on AAPL data.

