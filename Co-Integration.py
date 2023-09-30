import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

df = pd.read_csv('AssignmentV2_3.csv', usecols=["dates", "e", "d", "c"], parse_dates=['dates'], sep=';', decimal=',')
print(df.head())

# Set a custom style
plt.style.use('classic')

# Create a 1x3 grid of subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4), facecolor='white')

# Plot "e" on the first subplot and add a label
axes[0].plot(df['e'], color='red', linewidth=2, label='e')
axes[0].set_title("", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Quarters", fontsize=12)
axes[0].set_ylabel("Value", fontsize=12)
axes[0].grid(True)
axes[0].legend(loc='best')

# Plot "d" on the second subplot and add a label
axes[1].plot(df['d'], color='blue', linewidth=2, label='d')
axes[1].set_title("", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Quarters", fontsize=12)
axes[1].set_ylabel("Value", fontsize=12)
axes[1].grid(True)
axes[1].legend(loc='best')

# Plot "c" on the third subplot and add a label
axes[2].plot(df['c'], color='green', linewidth=2, label='c')
axes[2].set_title("", fontsize=14, fontweight='bold')
axes[2].set_xlabel("Quarters", fontsize=12)
axes[2].set_ylabel("Value", fontsize=12)
axes[2].grid(True)
axes[2].legend(loc='best')

# Add a common title to the whole figure
fig.suptitle('Time Series Plots for Variables e, d, and c', fontsize=16, fontweight='bold')

# Adjust spacing between subplots
fig.tight_layout()

plt.tight_layout()
plt.show()


#Making histograms
# Create a 1x3 grid of subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))

# Plot histogram of "e" with normal distribution curve on the first subplot and add a label
mu, std = df['e'].mean(), df['e'].std()
n, bins, patches = axes[0].hist(df['e'], bins=20, color='red', label='e', density=True)
y = ((1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * (1 / std * (bins - mu))**2))
axes[0].plot(bins, y, '--', color='black')
axes[0].set_title("Variable e", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Value", fontsize=12)
axes[0].set_ylabel("Density", fontsize=12)
axes[0].grid(True)
axes[0].legend(loc='best')

# Plot histogram of "d" with normal distribution curve on the second subplot and add a label
mu, std = df['d'].mean(), df['d'].std()
n, bins, patches = axes[1].hist(df['d'], bins=20, color='blue', label='d', density=True)
y = ((1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * (1 / std * (bins - mu))**2))
axes[1].plot(bins, y, '--', color='black')
axes[1].set_title("Variable d", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Value", fontsize=12)
axes[1].set_ylabel("Density", fontsize=12)
axes[1].grid(True)
axes[1].legend(loc='best')

# Plot histogram of "c" with normal distribution curve on the third subplot and add a label
mu, std = df['c'].mean(), df['c'].std()
n, bins, patches = axes[2].hist(df['c'], bins=20, color='green', label='c', density=True)
y = ((1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * (1 / std * (bins - mu))**2))
axes[2].plot(bins, y, '--', color='black')
axes[2].set_title("Variable c", fontsize=14, fontweight='bold')
axes[2].set_xlabel("Value", fontsize=12)
axes[2].set_ylabel("Density", fontsize=12)
axes[2].grid(True)
axes[2].legend(loc='best')

# Add a common title to the whole figure
fig.suptitle('Normal Probability Plots for Variables e, d, and c', fontsize=16, fontweight='bold')

# Adjust spacing between subplots
fig.tight_layout()

plt.tight_layout()
plt.show()

#Residual for e in histogram and qq plot
# Fit an OLS model to predict 'e' using 'd' and 'c'
model = sm.OLS(df['e'], sm.add_constant(df[['d', 'c']]))
results = model.fit()

# Calculate the residuals of the model
residuals = results.resid

# Plot the residuals against a normal distribution histogram
plt.hist(residuals, bins=20, density=True, alpha=0.7, color='blue', label='Residuals', histtype='step')

# Overlay a normal distribution on the histogram
mean = residuals.mean()
std = residuals.std()
x = np.linspace(mean - 3*std, mean + 3*std, 100)
y = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
plt.plot(x, y, '-r', label='Normal Distribution')

# Add labels and a legend to the plot
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Residuals vs. Normal Distribution')
plt.legend()
plt.show()

# Plot the residuals against a normal QQ plot
import scipy.stats as stats

stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Residuals QQ Plot")
plt.show()


# Calculate the descriptive statistics for each variable
e_desc = df['e'].describe()
d_desc = df['d'].describe()
c_desc = df['c'].describe()

# Combine the descriptive statistics into a single dataframe
desc_df = pd.concat([e_desc, d_desc, c_desc], axis=1)
desc_df.columns = ['e', 'd', 'c']

# Display the descriptive statistics
print(desc_df)

#GETS approach, while considering deterministic trend term

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.stattools import durbin_watson
import numpy as np

# Define a function to find the best AR model for each variable
def find_best_ar_model(variable_name, data, max_lags=10):
    best_model = None
    best_criterion_value = None
    best_trend = None
    best_lag = None
    trends = ['n', 'c', 't', 'ct']
    criterion = 'BIC'

    for trend in trends:
        aic_values, bic_values, hqic_values = [], [], []
        for i in range(1, max_lags + 1):
            model = AutoReg(data[variable_name], lags=i, trend=trend)
            result = model.fit()
            aic_values.append(result.aic)
            bic_values.append(result.bic)
            hqic_values.append(result.hqic)

        criterion_values = {
            'AIC': aic_values,
            'BIC': bic_values,
            'HQIC': hqic_values
        }
        current_best_lag = np.argmin(criterion_values[criterion]) + 1
        current_best_criterion_value = criterion_values[criterion][current_best_lag - 1]

        if best_criterion_value is None or current_best_criterion_value < best_criterion_value:
            best_criterion_value = current_best_criterion_value
            best_trend = trend
            best_lag = current_best_lag

    best_model = AutoReg(data[variable_name], lags=best_lag, trend=best_trend).fit()
    return best_model

# Analyze each variable separately
variables = ['e', 'd', 'c']
max_lags = 10  # You can change this value based on your desired maximum number of lags

for variable in variables:
    best_ar_model = find_best_ar_model(variable, df, max_lags)

    # Print the model summary
    print(f"Best AR model for {variable}:")
    print(best_ar_model.summary())
    print("\n---\n")

#The above says AR(1) while oxmetrics says AR(2) with er trend term, i think i will go for ox metrics
#####Using the gets approach to find a well-specified model

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.stattools import durbin_watson

# Load your data here as `df`

# Define a function to find the best AR model for each variable
def find_best_ar_model(variable_name, data, max_lags=10):
    aic_values, bic_values, hqic_values = [], [], []

    for i in range(1, max_lags + 1):
        model = AutoReg(data[variable_name], lags=i)
        result = model.fit()
        aic_values.append(result.aic)
        bic_values.append(result.bic)
        hqic_values.append(result.hqic)

    best_lags = {
        'AIC': np.argmin(aic_values) + 1,
        'BIC': np.argmin(bic_values) + 1,
        'HQIC': np.argmin(hqic_values) + 1
    }
    return best_lags

# Analyze each variable separately
variables = ['e', 'd', 'c']
max_lags = 10  # You can change this value based on your desired maximum number of lags

for variable in variables:
    best_lags = find_best_ar_model(variable, df, max_lags)
    print(f"Optimal number of lags for {variable} based on:")
    for criterion, lag in best_lags.items():
        print(f"  {criterion}: {lag}")

    # Choose the criterion and the corresponding best lag
    selected_criterion = 'BIC'  # Replace 'BIC' with the desired criterion
    selected_lag = best_lags[selected_criterion]

    # Fit the AR model with the optimal number of lags
    ar_model = AutoReg(df[variable], lags=selected_lag)
    ar_result = ar_model.fit()

    # Print the model summary
    print(ar_result.summary())

    # Check for autocorrelation using the Durbin-Watson statistic
    dw_stat = durbin_watson(ar_result.resid)
    print(f"Durbin-Watson statistic for {variable}: {dw_stat}")
    print("\n---\n")
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
  # Plot ACF and PACF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    plot_acf(df[variable], lags=20, ax=ax1)
    plot_pacf(df[variable], lags=20, ax=ax2)
    ax1.set_title(f"ACF for {variable}")
    ax2.set_title(f"PACF for {variable}")
    ax1.set_ylim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.5)
    plt.show()

    print("\n---\n")

####### GETS for all variables simultaneasly and AR model

from statsmodels.tsa.api import VAR

# Load your data here as `df`

# Choose the variables for which you want to find the best AR model
data = df[['e', 'd', 'c']]

# Find the optimal number of lags using AIC, BIC, and HQIC
max_lags = 10  # You can change this value based on your desired maximum number of lags
criteria = ['AIC', 'BIC', 'HQIC']
best_lags = {}

for criterion in criteria:
    lag_values = []
    for i in range(1, max_lags + 1):
        model = VAR(data)
        result = model.fit(i)
        lag_values.append(getattr(result, criterion.lower()))

    best_lag = np.argmin(lag_values) + 1
    best_lags[criterion] = best_lag
    print(f"Optimal number of lags for {criterion}: {best_lag}")

# Choose the criterion and the corresponding best lag
selected_criterion = 'BIC'  # Replace 'BIC' with the desired criterion
selected_lag = best_lags[selected_criterion]

# Fit the VAR model with the optimal number of lags
var_model = VAR(data)
var_result = var_model.fit(selected_lag)

# Print the model summary
print(var_result.summary())

# Check for autocorrelation using the Durbin-Watson statistic for each variable
dw_stats = durbin_watson(var_result.resid)
for i, col in enumerate(data.columns):
    print(f"Durbin-Watson statistic for {col}: {dw_stats[i]}")



#2 Unit root

variables = ['e', 'd', 'c']

for var in variables:
    result = adfuller(df[var], regression='ct')  # You can change the 'ct' to 'c' or 'nc' as needed
    print(f"\nADF test for {var}:")
    print(f"Test statistic: {result[0]:.3f}")
    print(f"P-value: {result[1]:.3f}")
    print(f"Critical values: {result[4]}")

    if result[1] < 0.05:
        print(f"The null hypothesis of a unit root in {var} can be rejected.")
    else:
        print(f"The null hypothesis of a unit root in {var} cannot be rejected.")


#Checking for unit root in e with a trend term

from statsmodels.tsa.stattools import adfuller

# Perform the ADF test on 'e' with a trend term
result = adfuller(df['e'], regression='ctt')

# Print the test statistic, p-value, and critical values
print(f"ADF test for 'e' with a trend term:")
print(f"Test statistic: {result[0]:.3f}")
print(f"P-value: {result[1]:.3f}")
print(f"Critical values: {result[4]}")

# Check if the null hypothesis of a unit root can be rejected or not
if result[1] < 0.05:
    print("The null hypothesis of a unit root in 'e' with a trend term can be rejected.")
else:
    print("The null hypothesis of a unit root in 'e' with a trend term cannot be rejected.")

#3 Engle-Granger test

# Assuming you have 'e', 'd', and 'c' in your DataFrame 'df'
# Estimate the long-run relationship using OLS
y = df['e']  # Dependent variable
X = df[['d', 'c']]  # Independent variables
X = sm.add_constant(X)  # Add a constant term

model = sm.OLS(y, X).fit()
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

#4 Check for cointegration based on the ADF test result
if adf_stat < adf_critical_values['1%']:
    print('The series are cointegrated at 1% significance level.')
elif adf_stat < adf_critical_values['5%']:
    print('The series are cointegrated at 5% significance level.')
elif adf_stat < adf_critical_values['10%']:
    print('The series are cointegrated at 10% significance level.')
else:
    print('The series are not cointegrated.')


#5 Now ECM

# Calculate first differences
df_diff = df.diff().dropna()

# Estimate the long-run relationship (cointegrating equation) using OLS
y = df['e']
X = df[['d', 'c']]
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
residuals = model.resid

# Add the error correction term (ECT) to the differenced data
df_diff['ECT'] = residuals.shift(1)
df_diff = df_diff.dropna()

# Estimate the ECM with the first differences and the ECT
y_ecm = df_diff['e']
X_ecm = df_diff[['d', 'c', 'ECT']]
X_ecm = sm.add_constant(X_ecm)

model_ecm = sm.OLS(y_ecm, X_ecm).fit()
print(model_ecm.summary())

# To capture better, we can set up lags in th

n_lags = 1 # you can change this to include more lags

for lag in range(1, n_lags + 1):
    df_diff[f'delta_d_lag{lag}'] = df_diff['d'].shift(lag)
    df_diff[f'delta_c_lag{lag}'] = df_diff['c'].shift(lag)

df_diff.dropna(inplace=True)
y_ecm = df_diff['e']
X_ecm = df_diff[['d', 'c', 'ECT'] + [f'delta_d_lag{i}' for i in range(1, n_lags + 1)] + [f'delta_c_lag{i}' for i in range(1, n_lags + 1)]]
X_ecm = sm.add_constant(X_ecm)

model_ecm = sm.OLS(y_ecm, X_ecm).fit()
print(model_ecm.summary())

#REmoving insignifcant lags and estimating again

X_ecm_updated = df_diff[['d', 'c', 'ECT']]
X_ecm_updated = sm.add_constant(X_ecm_updated)

model_ecm_updated = sm.OLS(y_ecm, X_ecm_updated).fit()
print(model_ecm_updated.summary())

n_lags = 1 # you can change this to include more lags

for lag in range(1, n_lags + 1):
    df_diff[f'delta_d_lag{lag}'] = df_diff['d'].shift(lag)
    df_diff[f'delta_c_lag{lag}'] = df_diff['c'].shift(lag)

df_diff.dropna(inplace=True)
y_ecm = df_diff['e']
X_ecm = df_diff[['d', 'c', 'ECT'] + [f'delta_d_lag{i}' for i in range(1, n_lags + 1)] + [f'delta_c_lag{i}' for i in range(1, n_lags + 1)]]
X_ecm = sm.add_constant(X_ecm)

model_ecm = sm.OLS(y_ecm, X_ecm).fit()
print(model_ecm.summary())

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
