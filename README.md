# Stock Forecasting Project

## Overview
This project uses various time series forecasting models to predict stock prices. The models used are:
- **ARIMA (AutoRegressive Integrated Moving Average)**
- **Holt-Winters Exponential Smoothing**
- **Prophet by Facebook**

The goal of the project is to forecast future stock prices based on historical stock data, using the mentioned models, and to evaluate their accuracy using **Mean Squared Error (MSE)**.

## Features
- Time series forecasting using ARIMA, Holt-Winters, and Prophet models.
- Data preprocessing (handling missing values, stationarity checks, outlier removal).
- Forecasting stock prices for the next 3650 business days (approximately 14 years).
- Model evaluation using Mean Squared Error (MSE).
- Visualization of original stock data vs. forecasted values.

## Requirements
Make sure you have Python 3.6+ installed. The required libraries are:
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `statsmodels`
- `prophet`
- `sklearn`

You can install the required libraries by running:

```bash
pip install -r requirements.txt
