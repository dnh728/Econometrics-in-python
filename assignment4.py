import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the data
df = pd.read_csv('Assignment_4.csv', usecols=["Date", "DAX", "DlogDAX", "y1", "y2", "y3"], sep=';', decimal=',')
df = df.replace(np.nan, 0)

# Create lags for y1_t, y2_t, and y3_t
for i in range(1, 5):
    df[f'y1_t_lag{i}'] = df['y1'].shift(i)
    df[f'y2_t_lag{i}'] = df['y2'].shift(i)
    df[f'y3_t_lag{i}'] = df['y3'].shift(i)

# Drop the first four rows with missing lag values
df = df.dropna()

# Split the data into endogenous (DAX log returns) and exogenous variables (lags of y1_t, y2_t, and y3_t)
endog = df['DlogDAX']
exog = df[['y1_t_lag1', 'y1_t_lag2', 'y1_t_lag3', 'y1_t_lag4', 'y2_t_lag1', 'y2_t_lag2', 'y2_t_lag3', 'y2_t_lag4', 'y3_t_lag1', 'y3_t_lag2', 'y3_t_lag3', 'y3_t_lag4']]

# Fit GARCH model with exogenous variables and 4 lags
garch_model_with_exog = arch_model(endog, x=exog, vol='Garch', p=1, q=1, dist='Normal')
garch_result_with_exog = garch_model_with_exog.fit()

print("GARCH Model with Exogenous Variables and 4 Lags:")
print(garch_result_with_exog.summary())