import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plot
from arch.univariate import arch_model

from preliminary import (data)

def exceedence(data):
    L = len(data)
    Loss = data['Loss']

    exceed_1 = []
    exceed_2 = []
    exceed_3 = []
    exceed_4 = []

    column_1 = data['Estimated VaR 95%']
    column_2 = data['Estimated VaR 99%']
    column_3 = data['Estimated ES 95%']
    column_4 = data['Estimated ES 99%']

    for i in range(L):
        if i < 501:
            exceed_1.append(0)
            exceed_2.append(0)
            exceed_3.append(0)
            exceed_4.append(0)
        else:
            if Loss[i] > column_1[i]:
                exceed_1.append(data['Loss'][i])
            else:
                exceed_1.append(0)

            if Loss[i] > column_2[i]:
                exceed_2.append(data['Loss'][i])
            else:
                exceed_2.append(0)

            if Loss[i] > column_3[i]:
                exceed_3.append(data['Loss'][i])
            else:
                exceed_3.append(0)

            if Loss[i] > column_4[i]:
                exceed_4.append(data['Loss'][i])
            else:
                exceed_4.append(0)

    data['Exceedence VaR 95%'] = exceed_1
    data['Exceedence VaR 99%'] = exceed_2
    data['Exceedence ES 95%'] = exceed_3
    data['Exceedence ES 99%'] = exceed_4

    VaR_95_violations = len(exceed_1) - exceed_1.count(0)
    VaR_99_violations = len(exceed_2) - exceed_2.count(0)
    ES_95_violations = len(exceed_3) - exceed_3.count(0)
    ES_99_violations = len(exceed_4) - exceed_4.count(0)

    return [data, VaR_95_violations, VaR_99_violations, ES_95_violations, ES_99_violations]

def EWMA(mu, alpha, sigma_0, data, column):

    L = len(data)
    X = data[column]
    sigma = [sigma_0]

    if L != 1:
        sigma_n_minus_1 = sigma_0
        X_n_minus_1 = X[0]

        for i in range(1, L):
            sigma_n = (alpha * ((X_n_minus_1 - mu)**2)) + ((1 - alpha) * (sigma_n_minus_1))

            sigma.append(math.sqrt(sigma_n))

            sigma_n_minus_1 = sigma_n
            X_n_minus_1 = X[i]

    data['EWMA'] = sigma
            
    return data

def GARCH_FHS(mu, alpha, sigma_0, data, column):

    L = len(data)
    X = data[column]

    estimated_VaR_95 = []
    estimated_VaR_99 = []
    estimated_ES_95 = []
    estimated_ES_99 = []

    data_GARCH = GARCH_VaR_ES(data, 'Loss', 500)

    for i in range(L):
        estimated_VaR_95.append(mu + (data_GARCH['var'][i] * data_GARCH['VaR 95%'][i]))
        estimated_VaR_99.append(mu + (data_GARCH['var'][i] * data_GARCH['VaR 99%'][i]))
        estimated_ES_95.append(mu + (data_GARCH['var'][i] * data_GARCH['ES 95%'][i]))
        estimated_ES_99.append(mu + (data_GARCH['var'][i] * data_GARCH['ES 99%'][i]))

    df_estimates = pd.DataFrame()
    
    df_estimates['Date'] = data['Date']
    df_estimates['Loss'] = data['Loss']
    df_estimates['Estimated VaR 95%'] = estimated_VaR_95
    df_estimates['Estimated VaR 99%'] = estimated_VaR_99
    df_estimates['Estimated ES 95%'] = estimated_ES_95
    df_estimates['Estimated ES 99%'] = estimated_ES_99

    exceed = exceedence(df_estimates)
    data = exceed[0]
    VaR_95_violations = exceed[1]
    VaR_99_violations = exceed[2]
    ES_95_violations = exceed[3]
    ES_99_violations = exceed[4]

    return [df_estimates, VaR_95_violations, VaR_99_violations, ES_95_violations, ES_99_violations]

def GARCH_VaR_ES(data, column, lag):

    C = len(data)
    quantiles_95 = []
    quantiles_99 = []
    es_95 = []
    es_99 = []
    var = []

    for i in range(C):
        if i < lag:
            quantiles_95.append(0)
            quantiles_99.append(0)
            es_95.append(0)
            es_99.append(0)
            var.append(0)
        else:
            model = arch_model(data['Loss'][i - (lag):i], mean='zero', p=1, q=1, dist='normal')
            res = model.fit(update_freq=5, disp=0)
            forecasts = res.forecast(reindex=False)
            standardised_values = res.std_resid
            var_val = forecasts.variance.iloc[0,0]
            q_95 = np.quantile(standardised_values.values, 0.95)
            q_99 = np.quantile(standardised_values.values, 0.99)
            quantiles_95.append(q_95)
            quantiles_99.append(q_99)
            var.append(math.sqrt(var_val))

            count_95 = 0
            count_99 = 0

            es_sum_95 = 0
            es_sum_99 = 0

            for j in range(0, lag):
                if standardised_values.values[j] >= q_95:
                    count_95 += 1
                    es_sum_95 += standardised_values.values[j]
                if standardised_values.values[j] >= q_99:
                    count_99 += 1
                    es_sum_99 += standardised_values.values[j]
            
            if count_95 != 0:
                es_95.append((1 / count_95) * es_sum_95)
            else:
                es_95.append(0)
            if count_99 != 0:
                es_99.append((1 / count_99) * es_sum_99)
            else:
                es_99.append(0)
    
    data['var'] = var
    data['VaR 95%'] = quantiles_95
    data['VaR 99%'] = quantiles_99
    data['ES 95%'] = es_95
    data['ES 99%'] = es_99

    return data

def positive_loss(data, lag, column):
    data_positive_loss = pd.DataFrame()
    for index, row in data.iterrows():
        if row[column] > 0 and index > lag:
            data_positive_loss = data_positive_loss.append(row)

    return data_positive_loss

start_df = GARCH_FHS(0, 0.06, data['Loss'][0], data, 'Loss')
df = positive_loss(start_df[0], 500, 'Loss')
print(start_df[0])
print(start_df[1])
print(start_df[2])

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
