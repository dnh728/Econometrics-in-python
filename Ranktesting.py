import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

df = pd.read_csv('AssignmentV2_3.csv', usecols=["dates", "e", "d", "c"], parse_dates=['dates'], sep=';', decimal=',')
print(df.head())

def adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")

for column in df.columns[1:]:
    print(f"ADF test for {column}:")
    adf_test(df[column])
    print("\n")

def adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")

for column in df.columns[1:]:
    print(f"ADF test for {column}:")
    adf_test(df[column])
    print("\n")

def johansen_test(data, p):
    result = coint_johansen(data, det_order=0, k_ar_diff=p)
    print("Eigenvalues:\n", result.eig)
    print("\nTrace statistics:\n", result.lr1)
    print("\nCritical values (90%, 95%, 99%) of trace statistic:\n", result.cvt)
    print("\nMax eigen statistics:\n", result.lr2)
    print("\nCritical values (90%, 95%, 99%) of max eigen statistic:\n", result.cvm)

# Perform the Johansen test for VECM(1)
johansen_test(df[['e', 'd', 'c']], 1)

# Prepare the data
data = df[["e", "d", "c"]].values

# Estimate unrestricted VECM(1) model with rank 1
model_unrestricted = VECM(data, k_ar_diff=1, coint_rank=1, deterministic="ci")
results_unrestricted = model_unrestricted.fit()

# Estimate restricted VECM(1) model with rank 1
# Restrict β[1] to be 1
restricted_beta = np.array([[1], [-1], [-1.18]])
model_restricted = VECM(data, k_ar_diff=1, coint_rank=1, deterministic="ci", beta=restricted_beta)
results_restricted = model_restricted.fit()

# Organize results in a table
alpha_unrestricted = results_unrestricted.alpha
t_values_alpha_unrestricted = results_unrestricted.tvalues_alpha
beta_unrestricted = results_unrestricted.beta
t_values_beta_unrestricted = results_unrestricted.tvalues_beta

alpha_restricted = results_restricted.alpha
t_values_alpha_restricted = results_restricted.tvalues_alpha
beta_restricted = results_restricted.beta
t_values_beta_restricted = results_restricted.tvalues_beta

table_4 = pd.DataFrame({
    "Unrestricted": [f"{alpha_unrestricted[0,0]:.3f} [{t_values_alpha_unrestricted[0,0]:.1f}]",
                     f"{alpha_unrestricted[1,0]:.3f} [{t_values_alpha_unrestricted[1,0]:.1f}]",
                     f"{alpha_unrestricted[2,0]:.3f} [{t_values_alpha_unrestricted[2,0]:.1f}]",
                     f"{beta_unrestricted[0,0]:.0f}",
                     f"{beta_unrestricted[1,0]:.3f} [{t_values_beta_unrestricted[1,0]:.1f}]",
                     f"{beta_unrestricted[2,0]:.3f} [{t_values_beta_unrestricted[2,0]:.1f}]"],
    "Restricted": [f"{alpha_restricted[0,0]:.3f} [{t_values_alpha_restricted[0,0]:.1f}]",
                   f"{alpha_restricted[1,0]:.4f} [{t_values_alpha_restricted[1,0]:.1f}]",
                   f"{alpha_restricted[2,0]:.3f} [{t_values_alpha_restricted[2,0]:.1f}]",
                   f"{beta_restricted[0,0]:.0f}",
                   "",
                   f"{beta_restricted[2,0]:.2f} [{t_values_beta_restricted[2,0]:.1f}]"]
}, index=["α_e [t-value]", "α_d [t-value]", "α_c [t-value]", "β_e [t-value]", "β_d [t-value]", "β_c [t-value]"])

print("Table 4: Co-integrated model parameters")
print(table_4)

# Test the hypothesis that the long-run demand elasticity is β_1=1
hypothesis_test = results_unrestricted.test_beta_eq([1, 0, 0], value=[0, 1, 0])
print("\nTable 5: Hypothesis Test")
print(hypothesis_test.summary())

# The co-integration relationship
print("\nCo-integration relationship:")
print(f"e_t = {beta_unrestricted[1,0]:.3f}d_t + {beta_unrestricted[2,0]:.3f}c_t")