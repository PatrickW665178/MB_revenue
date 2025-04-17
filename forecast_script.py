# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 22:10:14 2025

@author: patri
"""
#%%
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.stattools import adfuller, kpss, pacf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR as VAR_api

#%%
# Equivalent of reading and transforming data
data = pd.read_csv("MB_revenue.csv", header=0)
data = data.iloc[:48, 1:]

# Equivalent of creating time series objects
desktop = pd.Series(data.iloc[:, 5].values, index=pd.to_datetime(pd.date_range(start='2009-01-01', periods=48, freq='QS')))
laptop = pd.Series(data.iloc[:, 6].values, index=pd.to_datetime(pd.date_range(start='2009-01-01', periods=48, freq='QS')))
server = pd.Series(data.iloc[:, 7].values, index=pd.to_datetime(pd.date_range(start='2009-01-01', periods=48, freq='QS')))
mb = pd.Series(data.iloc[:, 10].values, index=pd.to_datetime(pd.date_range(start='2009-01-01', periods=48, freq='QS')))

tablex = pd.DataFrame(np.zeros((3, 3)), index=["RMSE", "MAE", "MAPE"], columns=["desktop", "laptop", "server"])
table_mb = pd.DataFrame(np.zeros((3, 4)), index=["RMSE", "MAE", "MAPE"], columns=["SARIMA", "DARIMA", "Lag", "VAR"])
#%%
###### MB shipment forecasting using Holt-Winters smoothing#舊版
# additive Holt-Winters smoothing (含趨勢與季節性，皆用加法)
fit_mb1 = ExponentialSmoothing(mb, trend='add', seasonal='add', 
                               seasonal_periods=4, initialization_method="estimated").fit()

# multiplicative Holt-Winters smoothing (含趨勢與季節性，皆使用乘法)
print("Minimum value before shift:", mb.min())
mb_cleaned = mb.dropna()
fit_mb2 = ExponentialSmoothing(mb_cleaned, trend='mul', seasonal='mul', 
                               seasonal_periods=4, initialization_method="estimated").fit()

# without trend: 只使用季節性模型 (無趨勢成分)
fit_mb3 = ExponentialSmoothing(mb, trend=None, seasonal='add', 
                               seasonal_periods=4, initialization_method="estimated").fit()

# non-seasonal: 簡單指數平滑 (不含季節性)
fit_mb4 = SimpleExpSmoothing(mb).fit()

plt.figure(figsize=(10, 6))
plt.plot(mb, label="Actual", color="black")
plt.plot(fit_mb1.fittedvalues, label="H-W additive", color="red")
plt.plot(fit_mb2.fittedvalues, label="H-W multiplicative", color="green")
plt.plot(fit_mb3.fittedvalues, label="Without trend", color="blue")
plt.plot(fit_mb4.fittedvalues, label="Non-seasonal", color="orange")
plt.title("Forecasts for MB")
plt.xlabel("Season")
plt.ylabel("Shipments")
plt.legend(loc="upper right", fontsize="small")
plt.show()

# 建立存放誤差評估結果的表格
table = pd.DataFrame(np.zeros((3, 4)), 
                     index=["RMSE", "MAE", "MAPE"],
                     columns=["additive", "multiplicative", "without trend", "non-seasonal"])
#%%

# RMSE
table.loc["RMSE", "additive"] = np.sqrt(mean_squared_error(mb, fit_mb1.fittedvalues))
table.loc["RMSE", "multiplicative"] = np.sqrt(mean_squared_error(mb_cleaned, fit_mb2.fittedvalues))
table.loc["RMSE", "without trend"] = np.sqrt(mean_squared_error(mb, fit_mb3.fittedvalues))
table.loc["RMSE", "non-seasonal"] = np.sqrt(mean_squared_error(mb, fit_mb4.fittedvalues))

# MAE
table.loc["MAE", "additive"] = mean_absolute_error(mb, fit_mb1.fittedvalues)
table.loc["MAE", "multiplicative"] = mean_absolute_error(mb_cleaned, fit_mb2.fittedvalues)
table.loc["MAE", "without trend"] = mean_absolute_error(mb, fit_mb3.fittedvalues)
table.loc["MAE", "non-seasonal"] = mean_absolute_error(mb, fit_mb4.fittedvalues)

# MAPE
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

table.loc["MAPE", "additive"] = mape(mb, fit_mb1.fittedvalues)
table.loc["MAPE", "multiplicative"] = mape(mb_cleaned, fit_mb2.fittedvalues)
table.loc["MAPE", "without trend"] = mape(mb, fit_mb3.fittedvalues)
table.loc["MAPE", "non-seasonal"] = mape(mb, fit_mb4.fittedvalues)

print("Error matrix for MB (Holt-Winters):\n", table)

print("Forecasts for MB (Holt-Winters additive):\n", fit_mb1.forecast(4))
print("Forecasts for MB (Holt-Winters multiplicative):\n", fit_mb2.forecast(4))
print("Forecasts for MB (Holt-Winters without trend):\n", fit_mb3.forecast(4))
print("Forecasts for MB (Non-seasonal):\n", fit_mb4.forecast(4))
#%%
##################
####stationary####
##################
# Equivalent of ndiffs in R - using ADF test for number of differences
def n_diffs(series):
    adf_result = adfuller(series, autolag='AIC')
    if adf_result[1] > 0.05:
        return 1
    else:
        return 0

print("Number of differences needed:")
print("Desktop:", n_diffs(desktop))
print("Laptop:", n_diffs(laptop))
print("Server:", n_diffs(server))
print("MB:", n_diffs(mb))

# Desktop
print("\nDesktop Stationarity Tests:")
print("ADF Test:", adfuller(desktop))
print("KPSS Test:", kpss(desktop, nlags='short'))

d1_x6 = desktop.diff().dropna()
print("\nDesktop Stationarity Tests (after 1st difference):")
print("ADF Test:", adfuller(d1_x6))
print("KPSS Test:", kpss(d1_x6, nlags='short'))

# Laptop
print("\nLaptop Stationarity Tests:")
print("ADF Test:", adfuller(laptop))
print("KPSS Test:", kpss(laptop, nlags='short'))

d1_x7 = laptop.diff().dropna()
print("\nLaptop Stationarity Tests (after 1st difference):")
print("ADF Test:", adfuller(d1_x7))
print("KPSS Test:", kpss(d1_x7, nlags='short'))

# Server
print("\nServer Stationarity Tests:")
print("ADF Test:", adfuller(server))
print("KPSS Test:", kpss(server, nlags='short'))

d1_x8 = server.diff().dropna()
print("\nServer Stationarity Tests (after 1st difference):")
print("ADF Test:", adfuller(d1_x8))
print("KPSS Test:", kpss(d1_x8, nlags='short'))

# Motherboard
print("\nMotherboard Stationarity Tests:")
print("ADF Test:", adfuller(mb))
print("KPSS Test:", kpss(mb, nlags='short'))

d1_y = mb.diff().dropna()
print("\nMotherboard Stationarity Tests (after 1st difference):")
print("ADF Test:", adfuller(d1_y))
print("KPSS Test:", kpss(d1_y, nlags='short'))
#%%
##################
### PC/Desktop ###
##################
# Equivalent of auto.arima - using manual ARIMA based on R output
arimax6 = SARIMAX(desktop, order=(0, 1, 0), seasonal_order=(1, 1, 0, 4), trend='c').fit()
print("\nSummary of arimax6 (Desktop):\n", arimax6.summary())

lb_test_x6 = acorr_ljungbox(arimax6.resid, lags=[4], return_df=True)
print("\nLjung-Box Test for arimax6 (Desktop):\n", lb_test_x6)

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].plot(arimax6.resid)
axes[0].set_ylabel('Residuals')
plot_acf(arimax6.resid, lags=20, ax=axes[1])
plot_pacf(arimax6.resid, lags=20, ax=axes[2])
plt.tight_layout()
plt.show()

tablex.loc["RMSE", "desktop"] = np.sqrt(mean_squared_error(desktop[1:], arimax6.fittedvalues)) # MSE
tablex.loc["MAE", "desktop"] = mean_absolute_error(desktop[1:], arimax6.fittedvalues) # MAE
tablex.loc["MAPE", "desktop"] = mape(desktop[1:], arimax6.fittedvalues) # MAPE

# Forecasting desktop
f_x6 = arimax6.get_forecast(steps=4)
print("\nForecast for desktop:\n", f_x6.summary_frame(alpha=0.05))

plt.figure(figsize=(10, 6))
plt.plot(desktop, label="Real", color="red")
plt.plot(arimax6.fittedvalues, label="Predicted", color="green")
plt.plot(f_x6.predicted_mean, label="Forecast", color="blue")
plt.title("Forecast for desktop")
plt.xlabel("Year")
plt.ylabel("Shipment (thousands)")
plt.legend(loc="bottomleft")
plt.show()
#%%
#################
### NB/Laptop ###
#################
# Equivalent of auto.arima - using manual ARIMA based on R output
arimax7 = SARIMAX(laptop, order=(1, 0, 0), seasonal_order=(0, 1, 0, 4), trend='c').fit()
print("\nSummary of arimax7 (Laptop):\n", arimax7.summary())

lb_test_x7 = acorr_ljungbox(arimax7.resid, lags=[4], return_df=True)
print("\nLjung-Box Test for arimax7 (Laptop):\n", lb_test_x7)

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].plot(arimax7.resid)
axes[0].set_ylabel('Residuals')
plot_acf(arimax7.resid, lags=20, ax=axes[1])
plot_pacf(arimax7.resid, lags=20, ax=axes[2])
plt.tight_layout()
plt.show()

tablex.loc["RMSE", "laptop"] = np.sqrt(mean_squared_error(laptop, arimax7.fittedvalues)) # MSE
tablex.loc["MAE", "laptop"] = mean_absolute_error(laptop, arimax7.fittedvalues) # MAE
tablex.loc["MAPE", "laptop"] = mape(laptop, arimax7.fittedvalues) # MAPE

# Forecasting laptop
f_x7 = arimax7.get_forecast(steps=4)
print("\nForecast for laptop:\n", f_x7.summary_frame(alpha=0.05))

plt.figure(figsize=(10, 6))
plt.plot(laptop, label="Real", color="red")
plt.plot(arimax7.fittedvalues, label="Predicted", color="green")
plt.plot(f_x7.predicted_mean, label="Forecast", color="blue")
plt.title("Forecast for laptop")
plt.xlabel("Year")
plt.ylabel("Shipment (thousands)")
plt.legend(loc="topleft")
plt.show()
#%%
##############
### server ###
##############
# Equivalent of auto.arima - using manual ARIMA based on R output
arimax8 = ARIMA(server, order=(0, 1, 1), trend='c').fit() # better
print("\nSummary of arimax8 (Server):\n", arimax8.summary())

lb_test_x8 = acorr_ljungbox(arimax8.resid, lags=[4], return_df=True)
print("\nLjung-Box Test for arimax8 (Server):\n", lb_test_x8)

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].plot(arimax8.resid)
axes[0].set_ylabel('Residuals')
plot_acf(arimax8.resid, lags=20, ax=axes[1])
plot_pacf(arimax8.resid, lags=20, ax=axes[2])
plt.tight_layout()
plt.show()

tablex.loc["RMSE", "server"] = np.sqrt(mean_squared_error(server[1:], arimax8.fittedvalues)) # MSE
tablex.loc["MAE", "server"] = mean_absolute_error(server[1:], arimax8.fittedvalues) # MAE
tablex.loc["MAPE", "server"] = mape(server[1:], arimax8.fittedvalues) # MAPE

# Forecasting server
f_x8 = arimax8.get_forecast(steps=4)
print("\nForecast for server:\n", f_x8.summary_frame(alpha=0.05))

plt.figure(figsize=(10, 6))
plt.plot(server, label="Real", color="red")
plt.plot(arimax8.fittedvalues, label="Predicted", color="green")
plt.plot(f_x8.predicted_mean, label="Forecast", color="blue")
plt.title("Forecast for server")
plt.xlabel("Year")
plt.ylabel("Shipment (thousands)")
plt.legend(loc="topleft")
plt.show()

print("\nError matrix for three computer products:\n", tablex) # three computer products
#%%
###############
###### mb #####
###############
# Equivalent of auto.arima - using manual ARIMA based on R output
arimay = SARIMAX(mb, order=(2, 1, 0), seasonal_order=(2, 0, 0, 4), trend='c').fit()
print("\nSummary of arimay (MB):\n", arimay.summary())

lb_test_y = acorr_ljungbox(arimay.resid, lags=[4], return_df=True)
print("\nLjung-Box Test for arimay (MB):\n", lb_test_y)

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].plot(arimay.resid)
axes[0].set_ylabel('Residuals')
plot_acf(arimay.resid, lags=20, ax=axes[1])
plot_pacf(arimay.resid, lags=20, ax=axes[2])
plt.tight_layout()
plt.show()

table_mb.loc["RMSE", "SARIMA"] = np.sqrt(mean_squared_error(mb[1:], arimay.fittedvalues)) # RMSE
table_mb.loc["MAE", "SARIMA"] = mean_absolute_error(mb[1:], arimay.fittedvalues) # MAE
table_mb.loc["MAPE", "SARIMA"] = mape(mb[1:], arimay.fittedvalues) # MAPE

# Forecasting mb
f_y = arimay.get_forecast(steps=4)
print("\nForecast for mb:\n", f_y.summary_frame(alpha=0.05))

plt.figure(figsize=(10, 6))
plt.plot(mb, label="Real", color="red")
plt.plot(arimay.fittedvalues, label="Predicted", color="green")
plt.plot(f_y.predicted_mean, label="Forecast", color="blue")
plt.title("Forecast for mb")
plt.xlabel("Year")
plt.ylabel("Shipment (thousands)")
plt.legend(loc="bottomleft")
plt.show()
#%%
####################################################
####mb,desktop,laptop,server########################
####################################################
#####zero time lags
xmb_regressors = pd.DataFrame({'desktop': desktop, 'laptop': laptop, 'server': server}).set_index(mb.index)
arima_xmb = SARIMAX(mb, exog=xmb_regressors, order=(1, 0, 2), seasonal_order=(2, 0, 0, 4), trend='c').fit()
print("\nSummary of arima_xmb (MB with zero lags):\n", arima_xmb.summary())

lb_test_xmb_0 = acorr_ljungbox(arima_xmb.resid, lags=[4], return_df=True)
print("\nLjung-Box Test for arima_xmb (MB with zero lags):\n", lb_test_xmb_0)

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].plot(arima_xmb.resid)
axes[0].set_ylabel('Residuals')
plot_acf(arima_xmb.resid, lags=20, ax=axes[1])
plot_pacf(arima_xmb.resid, lags=20, ax=axes[2])
plt.tight_layout()
plt.show()

table_mb.loc["RMSE", "DARIMA"] = np.sqrt(mean_squared_error(mb, arima_xmb.fittedvalues)) # RMSE
table_mb.loc["MAE", "DARIMA"] = mean_absolute_error(mb, arima_xmb.fittedvalues) # MAE
table_mb.loc["MAPE", "DARIMA"] = mape(mb, arima_xmb.fittedvalues) # MAPE

# Forecasting with regressors
future_index = pd.to_datetime(pd.date_range(start=mb.index[-1] + pd.tseries.offsets.QuarterBegin(), periods=4, freq='QS'))
future_xmb_regressors = pd.DataFrame({
    'desktop': f_x6.predicted_mean.values,
    'laptop': f_x7.predicted_mean.values,
    'server': f_x8.predicted_mean.values
}, index=future_index)

f_xmb = arima_xmb.get_forecast(steps=4, exog=future_xmb_regressors)
print("\nForecast for motherboard without considering lags:\n", f_xmb.summary_frame(alpha=0.05))

plt.figure(figsize=(10, 6))
plt.plot(mb, label="Real", color="red")
plt.plot(arima_xmb.fittedvalues, label="Predicted", color="green")
plt.plot(f_xmb.predicted_mean, label="Forecast", color="blue")
plt.title("Forecast for motherboard without considering lags")
plt.xlabel("Year")
plt.ylabel("Shipment (thousands)")
plt.legend(loc="bottomleft")
plt.show()
#%%
############################################################
####mb,desktop,laptop,server################################
############################################################
best_aic = np.inf
best_i = 0
best_j = 0
best_k = 0

# This part requires careful implementation of lagged regressors in Python's SARIMAX
# Due to the complexity and the need for proper alignment, a direct equivalent requires more detailed handling.
# A simplified approach is shown below, but might not perfectly match the R code's logic.

max_lag = 4
mb_values = mb.values
desktop_values = desktop.values
laptop_values = laptop.values
server_values = server.values

for i in range(max_lag + 1):
    for j in range(max_lag + 1):
        for k in range(max_lag + 1):
            max_m = max(i, j, k)
            if len(mb_values) > max_m:
                mb_subset = mb_values[max_m:]
                desktop_subset = desktop_values[(max_m - i):(len(desktop_values) - i)]
                laptop_subset = laptop_values[(max_m - j):(len(laptop_values) - j)]
                server_subset = server_values[(max_m - k):(len(server_values) - k)]

                if len(mb_subset) == len(desktop_subset) == len(laptop_subset) == len(server_subset) > 0:
                    exog_lagged = pd.DataFrame({'desktop': desktop_subset, 'laptop': laptop_subset, 'server': server_subset}, index=mb.index[max_m:])
                    try:
                        model_lagged = SARIMAX(mb_subset, exog=exog_lagged, order=(1, 0, 4), seasonal_order=(2, 0, 0, 4), trend='c').fit(disp=False)
                        if model_lagged.aic < best_aic:
                            best_aic = model_lagged.aic
                            best_i = i
                            best_j = j
                            best_k = k
                    except Exception as e:
                        print(f"Error fitting with lags {i}, {j}, {k}: {e}")

print(f"\np={best_i}\t{best_j}\t{best_k}\tbest_aic {best_aic}")
# Based on R output: p= 1	0	4	best_aic ...
#%%
############################################################
####mb,desktop,laptop,server---AIC########################
############################################################
lag_desktop = 1
lag_laptop = 0
lag_server = 4

start_index = max(lag_desktop, lag_laptop, lag_server)
mb_aic = mb[start_index:]
desktop_aic = desktop[start_index - lag_desktop:len(desktop) - lag_desktop]
laptop_aic = laptop[start_index - lag_laptop:len(laptop) - lag_laptop]
server_aic = server[start_index - lag_server:len(server) - lag_server]

if len(mb_aic) == len(desktop_aic) == len(laptop_aic) == len(server_aic) > 0:
    xmb_aic_regressors = pd.DataFrame({'desktop': desktop_aic.values, 'laptop': laptop_aic.values, 'server': server_aic.values}, index=mb_aic.index)
    arima_aicmb = SARIMAX(mb_aic, exog=xmb_aic_regressors, order=(1, 0, 4), seasonal_order=(2, 0, 0, 4), trend='c').fit()
    print("\nSummary of arima_aicmb (MB with AIC lags):\n", arima_aicmb.summary())

    lb_test_aicmb = acorr_ljungbox(arima_aicmb.resid, lags=[4], return_df=True)
    print("\nLjung-Box Test for arima_aicmb (MB with AIC lags):\n", lb_test_aicmb)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes[0].plot(arima_aicmb.resid)
    axes[0].set_ylabel('Residuals')
    plot_acf(arima_aicmb.resid, lags=20, ax=axes[1])
    plot_pacf(arima_aicmb.resid, lags=20, ax=axes[2])
    plt.tight_layout()
    plt.show()

    table_mb.loc["RMSE", "Lag"] = np.sqrt(mean_squared_error(mb_aic, arima_aicmb.fittedvalues)) # RMSE
    table_mb.loc["MAE", "Lag"] = mean_absolute_error(mb_aic, arima_aicmb.fittedvalues) # MAE
    table_mb.loc["MAPE", "Lag"] = mape(mb_aic, arima_aicmb.fittedvalues) # MAPE

    # Forecasting with AIC lags
    future_index_aic = pd.to_datetime(pd.date_range(start=mb.index[-1] + pd.tseries.offsets.QuarterBegin(), periods=4, freq='QS'))
    future_desktop_lagged = np.concatenate([desktop[-lag_desktop:].values, f_x6.predicted_mean.values[:-lag_desktop]])
    future_laptop_lagged = f_x7.predicted_mean.values
    future_server_lagged = np.concatenate([server[-(lag_server):].values, f_x8.predicted_mean.values[:-lag_server]])

    future_xmb_aic_regressors = pd.DataFrame({
        'desktop': [desktop[-1]] + list(f_x6.predicted_mean[:3]), # Lag 1
        'laptop': f_x7.predicted_mean,                              # Lag 0
        'server': list(server[-4:])                               # Lag 4 - using actual last 4 values
    }, index=future_index_aic)

    f_xmb_aic = arima_aicmb.get_forecast(steps=4, exog=future_xmb_aic_regressors)
    print("\nForecast for motherboard considering AIC_lags:\n", f_xmb_aic.summary_frame(alpha=0.05))

    plt.figure(figsize=(10, 6))
    plt.plot(mb_aic, label="Real", color="red")
    plt.plot(arima_aicmb.fittedvalues, label="Predicted", color="green")
    plt.plot(f_xmb_aic.predicted_mean, label="Forecast", color="blue")
    plt.title("Forecast for motherboard considering AIC_lags")
    plt.xlabel("Year")
    plt.ylabel("Shipment (thousands)")
    plt.legend(loc="bottomleft")
    plt.show()
#%%
###########################################################
####mb,desktop,laptop,server##########VAR
###########################################################
xmb_df = pd.DataFrame({'mb': mb, 'desktop': desktop, 'laptop': laptop, 'server': server})

# Equivalent of VARselect - manual approach
from statsmodels.tsa.vector_ar.vecm import select_order
order_results = select_order(xmb_df, maxlags=3, deterministic="ct") # Equivalent of type="both"
print("\nVAR Order Selection (AIC):\n", order_results.aic)
var_lag = 4 # Using the lag specified in the R code

# varmodel
var_model = VAR_api(xmb_df)
var_results = var_model.fit(var_lag, trend='ct') # Equivalent of type="both"
print("\nSummary of VAR Model:\n", var_results.summary())

# Equivalent of stability test - using plots of roots
# 取得長達 10 期的脈衝反應分析 (IRF)
irf = var_results.irf(periods=10)

# 在 plot 時指定 impulse 和 response
irf.plot(impulse='mb', response='mb', orth=False)
plt.title("Stability of VAR Model (Impulse Response)")
plt.show()

#from statsmodels.tsa.vector_ar.irf import IRAnalysis
#irf = IRAnalysis(var_results, impulse='mb', response='mb', periods=10)
#irf.plot(orth=False, impulse_label='MB', response_label='MB')
#plt.title("Stability of VAR Model (Impulse Response)")
#plt.show()

var_predictions = var_results.forecast(var_results.model.endog[-var_lag:], steps=4)
print("\nVAR Model Predictions:\n", var_predictions)

resid_var = var_results.resid['mb']
table_mb.loc["RMSE", "VAR"] = np.sqrt(mean_squared_error(mb[var_lag:], var_results.fittedvalues['mb'])) # RMSE
table_mb.loc["MAE", "VAR"] = mean_absolute_error(mb[var_lag:], var_results.fittedvalues['mb']) # MAE
table_mb.loc["MAPE", "VAR"] = mape(mb[var_lag:], var_results.fittedvalues['mb']) # MAPE

print("\nError matrix for MB:\n", table_mb)
