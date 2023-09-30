import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

df = pd.read_csv('AssignmentV2_3.csv', usecols=["dates", "e", "d", "c"], parse_dates=['dates'], sep=';', decimal=',')
print(df.head())

plt.style.use('classic', facecolor='white')

#Dickey fuller test

#In OXmetrics i found out that via the GETS approach that the model is well-specified with AR(2) with a trend term.
#Now i will do the unit root test for AR(2) whit a trend term
# Perform the ADF test on variable 'e'
adf_test_result = adfuller(df['e'], maxlag=2, regression='ct', autolag=None)

# Print the results
print(f"ADF Statistic: {adf_test_result[0]}")
print(f"p-value: {adf_test_result[1]}")
print(f"Used lags: {adf_test_result[2]}")
print(f"Number of observations used: {adf_test_result[3]}")

for key, value in adf_test_result[4].items():
    print(f"Critical Value ({key}): {value}")

# Assuming 'e', 'd', and 'c' in your DataFrame 'df'
# Estimate the long-run relationship using OLS
y = df['e']  # Dependent variable
X = df[['d', 'c']]  # Independent variables
X = sm.add_constant(X)  # Add a constant term

model = sm.OLS(y, X).fit(cov_type='HC1')  # Specify the robust covariance matrix type; you can use 'HC0', 'HC1', 'HC2', or 'HC3'

residuals = model.resid

# Run the ADF test on the residuals
adf_result = adfuller(residuals, regression='ct')
adf_stat = adf_result[0]
adf_p_value = adf_result[1]
adf_critical_values = adf_result[4]

print(f'ADF Statistic: {adf_stat}')
print(f'p-value: {adf_p_value}')
print('Critical Values:')
for key, value in adf_critical_values.items():
    print(f'   {key}: {value}')

print(model.summary())

#4 Check for cointegration based on the ADF test result
if adf_stat < adf_critical_values['1%']:
    print('The series are cointegrated at 1% significance level.')
elif adf_stat < adf_critical_values['5%']:
    print('The series are cointegrated at 5% significance level.')
elif adf_stat < adf_critical_values['10%']:
    print('The series are cointegrated at 10% significance level.')
else:
    print('The series are not cointegrated.')

    # Plot the residuals
plt.figure(figsize=(10, 6), facecolor='white')
plt.plot(residuals)
plt.title('Residuals from OLS Regression')
plt.xlabel('Quarters')
plt.ylabel('Residuals')
plt.legend(loc='center')
plt.show()

#5 Now ECM for AR(2) with a trend term

# Calculate first differences
df_diff = df.diff().dropna()

# Estimate the long-run relationship (cointegrating equation) using OLS
y = df['e']
X = df[['d', 'c']]
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
residuals = model.resid

plt.show()

# Add the error correction term (ECT) to the differenced data
df_diff['ECT'] = residuals.shift(1)
df_diff = df_diff.dropna()

# Add AR(2) terms
df_diff['L1_e'] = df_diff['e'].shift(1)
df_diff['L2_e'] = df_diff['e'].shift(2)

# Add a trend term
df_diff['trend'] = np.arange(1, len(df_diff) + 1)

# Drop rows with NaN values due to the creation of lagged variables
df_diff = df_diff.dropna()

# Estimate the ECM with the first differences, the ECT, AR(2) terms, and a trend term
y_ecm = df_diff['e']
X_ecm = df_diff[['d', 'c', 'ECT', 'L1_e', 'L2_e', 'trend']]
X_ecm = sm.add_constant(X_ecm)

model_ecm = sm.OLS(y_ecm, X_ecm).fit()
print(model_ecm.summary())


# Plot the ECM's fitted values and the actual first differences of 'e'
fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
ax.plot(df_diff.index, df_diff['e'], label='First differences of e')
ax.plot(df_diff.index, model_ecm.fittedvalues, label='ECM Fitted Values', color='red')
ax.axhline(0, color='black', linestyle='--', lw=1)  # Add a horizontal line at zero
ax.legend()
ax.set_xlabel('Quarters')
plt.show()

# Check the ECT coefficient and its p-value from the ECM model summary:
ect_coefficient = model_ecm.params['ECT']
ect_p_value = model_ecm.pvalues['ECT']

print(f"ECT Coefficient: {ect_coefficient}")
print(f"ECT p-value: {ect_p_value}")

last_residual = residuals.iloc[-1]
print(f"Last residual value: {last_residual}")

# Chech if ECM is larger than the last residual
if ect_coefficient < last_residual:
    print('e is below its equilibrium level in the last period')
elif ect_coefficient > last_residual:
    print('e is above its equilibrium level in the last period')
else:
    print('e is not above nor below its equilibrium in the last period')

# Estimate ECM for 'd'
y_ecm_d = df_diff['d']
X_ecm_d = df_diff[['e', 'c', 'ECT', 'L1_e', 'L2_e', 'trend']]
X_ecm_d = sm.add_constant(X_ecm_d)

model_ecm_d = sm.OLS(y_ecm_d, X_ecm_d).fit()
print("ECM for 'd':")
print(model_ecm_d.summary())

# Estimate ECM for 'c'
y_ecm_c = df_diff['c']
X_ecm_c = df_diff[['e', 'd', 'ECT', 'L1_e', 'L2_e', 'trend']]
X_ecm_c = sm.add_constant(X_ecm_c)

model_ecm_c = sm.OLS(y_ecm_c, X_ecm_c).fit()
print("\nECM for 'c':")
print(model_ecm_c.summary())

#6: Recursive estimation
# Recursive estimation
min_sample_size = 10
n = len(df)

# Initialize a dictionary to store recursive coefficients and standard errors for each variable
recursive_coefficients = {'d': [], 'c': [], 'ECT': []}
recursive_se = {'d': [], 'c': [], 'ECT': []}

for t in range(min_sample_size, n):
    # Estimate the long-run relationship (cointegrating equation) using OLS
    y_temp = y.iloc[:t]
    X_temp = X.iloc[:t]
    model_temp = sm.OLS(y_temp, X_temp).fit()
    residuals_temp = model_temp.resid

    # Add the error correction term (ECT) to the differenced data
    df_diff_temp = df.iloc[:t].diff().dropna()
    df_diff_temp['ECT'] = residuals_temp.shift(1)
    df_diff_temp = df_diff_temp.dropna()

    # Estimate the ECM with the first differences and the ECT
    y_ecm_temp = df_diff_temp['e']
    X_ecm_temp = df_diff_temp[['d', 'c', 'ECT']]
    X_ecm_temp = sm.add_constant(X_ecm_temp)

    model_ecm_temp = sm.OLS(y_ecm_temp, X_ecm_temp).fit()

    # Append the coefficients and standard errors for each variable
    for var in recursive_coefficients:
        recursive_coefficients[var].append(model_ecm_temp.params[var])
        recursive_se[var].append(model_ecm_temp.bse[var])

# Plot the recursive coefficients for each variable with confidence bands
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
for idx, var in enumerate(recursive_coefficients):
    time_range = df.index[min_sample_size:]
    axes[idx].plot(time_range, recursive_coefficients[var], label=f'{var} Coefficient')
    axes[idx].fill_between(time_range, np.array(recursive_coefficients[var]) - 2 * np.array(recursive_se[var]), 
                           np.array(recursive_coefficients[var]) + 2 * np.array(recursive_se[var]), color='gray', alpha=0.5, label='95% Confidence Band')
    axes[idx].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[idx].set_title(f'Recursive {var} Coefficients with Confidence Bands')
    axes[idx].set_xlabel('Quarters')
    axes[idx].legend()

plt.tight_layout()
plt.show()


#7 dynamic multipliers
# Define the number of periods for the time horizon
time_horizon = 10

# Calculate dynamic multipliers
dynamic_multipliers = {'d': [1], 'c': [1], 'ECT': [1]}

for period in range(1, time_horizon):
    for var in ['d', 'c', 'ECT']:
        multiplier = 0
        for i in range(period):
            multiplier += model_ecm.params[var] ** i
        dynamic_multipliers[var].append(multiplier)

# Plot dynamic multipliers
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

for idx, var in enumerate(dynamic_multipliers):
    axes[idx].plot(range(time_horizon), dynamic_multipliers[var], label=f'{var} Multiplier')
    axes[idx].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[idx].set_title(f'Dynamic Multiplier for {var}')
    axes[idx].legend()

plt.tight_layout()
plt.show()

# Short-run effects
short_run_effects = {'d': model_ecm.params['d'], 'c': model_ecm.params['c']}

print("Short-run effects:")
for var, effect in short_run_effects.items():
    print(f"{var}: {effect:.4f}")


#Setting up a table for SHORT RUN VS. LONG RUN effects 

# Long-run effects
long_run_effects = {'d': model.params['d'], 'c': model.params['c']}

# Short-run effects
short_run_effects = {'d': model_ecm.params['d'], 'c': model_ecm.params['c']}

# Create a DataFrame to store the effects
effects_df = pd.DataFrame({'Short-run Effects': short_run_effects, 'Long-run Effects': long_run_effects})

print("Short-run and Long-run Effects:")
print(effects_df)


#Short-run Effects:

#For a 1% increase in international competitiveness (c), the size of the Danish industrial export market (e) increases by 0.0871% in the short run.
#For a 1% increase in Danish industrial exports in quantities (d), the size of the Danish industrial export market (e) increases by 0.3323% in the short run.

#Long-run Effects:

#For a 1% increase in international competitiveness (c), the size of the Danish industrial export market (e) increases by 0.6166% in the long run.
#For a 1% increase in Danish industrial exports in quantities (d), the size of the Danish industrial export market (e) increases by 0.8884% in the long run.


#Short run and Long run
# Set the number of periods (time horizons) for the dynamic multipliers
periods = 10

# Compute dynamic multipliers for each variable using the ECM model
dynamic_multipliers = np.zeros((periods, 2))
for t in range(periods):
    dynamic_multipliers[t, 0] = model_ecm.params['d'] * (1 if t == 0 else (1 - np.exp(-t / 4)))  # Example ECM: d * (1 - exp(-t / 4))
    dynamic_multipliers[t, 1] = model_ecm.params['c'] * (1 if t == 0 else (1 - np.exp(-t / 4)))  # Example ECM: c * (1 - exp(-t / 4))

# Plot the dynamic multipliers
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(periods), dynamic_multipliers[:, 0], label='Short-run solution (d)', marker='o')
ax.plot(range(periods), dynamic_multipliers[:, 1], label='Long-run solution (c)', marker='o')
ax.set_xlabel('Time horizon')
ax.set_ylabel('Dynamic multipliers')
ax.set_title('Dynamic Multipliers for Short-run and Long-run Solutions')
ax.legend()
plt.show()

#Bar that shows the short-run and long run effects

# Short-run and long-run effects for d and c
short_run_effects = {'d': 0.332321, 'c': 0.087099}
long_run_effects = {'d': 0.888367, 'c': 0.616565}

# Variables for labels and positions in the plot
variables = ['d', 'c']
x_pos = [0, 1]

# Plot the short-run and long-run solutions for d and c
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x_pos, [short_run_effects[var] for var in variables], width=0.4, label='Short-run Effects')
ax.bar([x + 0.4 for x in x_pos], [long_run_effects[var] for var in variables], width=0.4, label='Long-run Effects')

# Set labels and title
ax.set_xticks([x + 0.2 for x in x_pos])
ax.set_xticklabels(variables)
ax.set_ylabel('Effects')
ax.set_title('Short-run and Long-run Effects for d and c')
ax.legend()

plt.show()


#Latex format table
from tabulate import tabulate

short_run_effects = {'d': 0.332321, 'c': 0.087099}
long_run_effects = {'d': 0.888367, 'c': 0.616565}
variables = ['d', 'c']

table_data = [['Variable', 'Short-run Effects', 'Long-run Effects']]

for var in variables:
    table_data.append([var, short_run_effects[var], long_run_effects[var]])

table = tabulate(table_data, headers='firstrow', tablefmt='latex_booktabs')
print(table)

