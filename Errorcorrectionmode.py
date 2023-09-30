import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


data = pd.read_csv('AssignmentV2_3.csv', usecols=["dates", "e", "d", "c"], parse_dates=['dates'], sep=';', decimal=',')

#Printing relationship between variables/ covariance matrix
correlation_matrix = data.corr()

print(correlation_matrix)

# Assuming you have a pandas DataFrame named 'data' with your variables
X = data

# Add a constant term to the independent variables (if your model has an intercept)
X = sm.add_constant(X)

# Calculate the VIF for each variable
vif = pd.DataFrame()
vif["variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)

