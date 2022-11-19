import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plot

from preliminary import (data, C)
from CW1QU2A import (VaR_ES, positive_loss, HS)

def EWMA(mu, alpha, sigma_0, data, column):

    L = len(data)
    print(L)
    X = data[column]
    sigma = [sigma_0]
    unstandardised_residuals = []
    standardised_residuals = []

    if L != 1:
        sigma_n_minus_1 = sigma_0
        X_n_minus_1 = X[0]
        unstandardised_residuals.append(0)
        standardised_residuals.append(0)

        for i in range(1, L):
            sigma_n = (alpha * ((X_n_minus_1 - mu)**2)) + ((1 - alpha) * (sigma_n_minus_1))
            if sigma_n != 0:
                unstandardised_residual = X[i] - mu
                standardised_residual = (unstandardised_residual / math.sqrt(sigma_n))
            else:
                unstandardised_residual = 0
                standardised_residual = 0

            sigma.append(math.sqrt(sigma_n))
            unstandardised_residuals.append(unstandardised_residual)
            standardised_residuals.append(standardised_residual)

            sigma_n_minus_1 = sigma_n
            X_n_minus_1 = X[i]

    data['EWMA'] = sigma
    data['Unstandardised Residuals'] = unstandardised_residuals
    data['Standardised Residuals'] = standardised_residuals
            
    return data

print(EWMA(0, 0.06, data['Loss'][0], data, 'Loss'))

def FHS(mu, alpha, sigma_0, data, column):
    
    data = EWMA(mu, alpha, sigma_0, data, column)
    L = len(data)
    X = data[column]
    mu_SWN = 0
    sigma_SWN = 1 
    num_samples = L
    samples_SWN = np.random.normal(mu_SWN, sigma_SWN, size=num_samples)

    df_SWN = pd.DataFrame({'SWN': samples_SWN})
    data_SWN = VaR_ES(df_SWN, 'SWN', 500)

    estimated_VaR_95 = []
    estimated_VaR_99 = []
    estimated_ES_95 = []
    estimated_ES_99 = []

    df_estimates = pd.DataFrame()

    for i in range(L):
        estimated_VaR_95.append(mu + (data['EWMA'][i] * data_SWN['VaR 95%'][i]))
        estimated_VaR_99.append(mu + (data['EWMA'][i] * data_SWN['VaR 99%'][i]))
        estimated_ES_95.append(mu + (data['EWMA'][i] * data_SWN['ES 95%'][i]))
        estimated_ES_99.append(mu + (data['EWMA'][i] * data_SWN['ES 99%'][i]))
    
    df_estimates['Date'] = data['Date']
    df_estimates['Loss'] = data['Loss']
    df_estimates['Estimated VaR 95%'] = estimated_VaR_95
    df_estimates['Estimated VaR 99%'] = estimated_VaR_99
    df_estimates['Estimated ES 95%'] = estimated_ES_95
    df_estimates['Estimated ES 99%'] = estimated_ES_99

    return df_estimates

df = positive_loss(FHS(0, 0.06, data['Loss'][0], data, 'Loss'), 500, 'Loss')

ax = df[['Date', 'Loss']].plot(
    x='Date', kind='bar', color='orange')
df[['Date', 'Estimated VaR 95%']].plot.line(
    x='Date', linestyle='-', color='red', ax=ax)
df[['Date', 'Estimated VaR 99%']].plot.line(
    x='Date', linestyle='--', color='red', ax=ax)
df[['Date', 'Estimated ES 95%']].plot.line(
    x='Date', linestyle='-', color='blue', ax=ax)
df[['Date', 'Estimated ES 99%']].plot.line(
    x='Date', linestyle='--', color='blue', ax=ax)
plot.show()

df_residuals= HS(EWMA(0, 0.06, data['Loss'][0], data, 'Loss'), 'Standardised Residuals', 500)
print(df_residuals)

ax = df_residuals[['Date', 'Loss']].plot(
    x='Date', kind='bar', color='orange')
df_residuals[['Date', 'VaR 95%']].plot.line(
    x='Date', linestyle='-', color='red', ax=ax)
df_residuals[['Date', 'VaR 99%']].plot.line(
    x='Date', linestyle='--', color='red', ax=ax)
df_residuals[['Date', 'ES 95%']].plot.line(
    x='Date', linestyle='-', color='blue', ax=ax)
df_residuals[['Date', 'ES 99%']].plot.line(
    x='Date', linestyle='--', color='blue', ax=ax)
plot.show()
