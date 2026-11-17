# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:03:14 2026

@author: stewarta6
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

#pulling stock data from the alpaca market api
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

SYMBOL = "NVDA"
START_DATE = "2022-01-01"
END_DATE = "2026-11-12"

API_KEY = "PKYCJGUKLG55LG2BMTKLKA2Q3W"
API_SECRET = "CYa8sjDQYcuqKNjb6BXTHREisUwkdFGzdKY7PKjubdCf"
client = StockHistoricalDataClient(API_KEY, API_SECRET)

request_param = StockBarsRequest(
    symbol_or_symbols = SYMBOL,
    timeframe = TimeFrame.Day,
    start = START_DATE,
    end = END_DATE
    )
bars = client.get_stock_bars(request_param)
df = bars.df.reset_index()

df = df[df["symbol"]== SYMBOL]

#features for the model we created. how the machine can understand
#market conditions --> just trying to predit the next training days return
df["return"] = df["close"].pct_change()
df["volatility"]= df["return"].rolling(5).std()
df["ma_5"]= df["close"].rolling(5).mean()
df["ma_10"] = df["close"].rolling(10).mean()
df["ma_20"]= df["close"].rolling(20).mean()

df["target"] = df["close"].shift(-1)
df = df.dropna()

features = [
    "close",
    "return",
    "volatility",
    "ma_5",
    "ma_10",
    "ma_20",
    ]

#splitting data into two different sets: training and testing

x = df[features]
y = df["target"]

#Time-based split
split = int(len(df) * 0.8)

x_train, x_test = x.iloc[:split], x.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

#Making three different models


#random forest model
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth = 10,
    random_state=42
    )

rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
rf_mae = mean_absolute_error(y_test, rf_pred)

print(f"Random Forest MAE: ${rf_mae:.2f}")

#xgboost model

xgb = XGBRegressor(
    n_estimators = 500,
    max_depth= 5,
    learning_rate = 0.05,
    subsample=0.8,
    colsample_bytree= 0.8,
    random_state= 42
    )

xgb.fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)

print(f"XGBoost MAE: ${xgb_mae:.2f}")

#Linear Regression
linear = LinearRegression()
linear.fit(x_train, y_train)
linear_pred = linear.predict(x_test)
linear_mae = mean_absolute_error(y_test, linear_pred)

print(f"Linear Regression MAE: ${linear_mae:.2f}")



#plot comparison 

plt.figure(figsize=(14,7))

plt.plot(y_test.values, label= "Actual", linewidth=2)
plt.plot(rf_pred, label = "Random Forest")
plt.plot(xgb_pred, label = "XGBoost")
plt.plot(linear_pred, label="Linear Regression")

plt.title("NVDA Price Predition Model Comparison")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()