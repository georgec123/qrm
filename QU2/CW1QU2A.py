import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plot

from preliminary import (data)

def exceedence(data):
    L = len(data)
    Loss = data['Loss']

    exceed_1 = []
    exceed_2 = []
    exceed_3 = []
    exceed_4 = []

    column_1 = data['VaR 95%']
    column_2 = data['VaR 99%']
    column_3 = data['ES 95%']
    column_4 = data['ES 99%']

    for i in range(L):
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

    return data

def VaR_ES(data, column, lag):

    C = len(data)
    quantiles_95 = []
    quantiles_99 = []
    es_95 = []
    es_99 = []

    for i in range(C):
        if i < lag + 1:
            quantiles_95.append(0)
            quantiles_99.append(0)
            es_95.append(0)
            es_99.append(0)
        else:
            q_95 = np.quantile(data[column][i - (lag + 1):i - 1], 0.95)
            q_99 = np.quantile(data[column][i - (lag + 1):i - 1], 0.99)
            quantiles_95.append(q_95)
            quantiles_99.append(q_99)

            count_95 = 0
            count_99 = 0

            es_sum_95 = 0
            es_sum_99 = 0

            for j in range(i - (lag + 1), i - 1):
                if data[column][j] >= q_95:
                    count_95 += 1
                    es_sum_95 += data[column][j]
                if data[column][j] >= q_99:
                    count_99 += 1
                    es_sum_99 += data[column][j]

            es_95.append((1 / count_95) * es_sum_95)
            es_99.append((1 / count_99) * es_sum_99)


    data['VaR 95%'] = quantiles_95
    data['VaR 99%'] = quantiles_99
    data['ES 95%'] = es_95
    data['ES 99%'] = es_99

    print(data['Loss'][0] > data['VaR 95%'][0])
    
    data = exceedence(data)

    return data

def positive_loss(data, lag, column):
    data_positive_loss = pd.DataFrame()
    for index, row in data.iterrows():
        if row[column] > 0 and index > lag:
            data_positive_loss = data_positive_loss.append(row)
            
    return data_positive_loss

def HS(data, column, lag):
    data_VaR_ES = VaR_ES(data, column, lag)
    data_positive_loss = positive_loss(data_VaR_ES, lag, column)

    return data_positive_loss

data_positive_loss = HS(data, 'Loss', 500)

print(data_positive_loss)

ax = data_positive_loss[['Date', 'Loss']].plot(
    x='Date', kind='bar', color='orange')
data_positive_loss[['Date', 'Exceedence VaR 95%']].plot.scatter(
    x='Date', y='Exceedence VaR 95%', marker='x' ,color='red', ax=ax)
data_positive_loss[['Date', 'Exceedence VaR 99%']].plot.scatter(
    x='Date', y='Exceedence VaR 99%', marker='o', color='yellow', ax=ax)
data_positive_loss[['Date', 'Exceedence ES 95%']].plot.scatter(
    x='Date', y='Exceedence ES 95%', marker=',', color='blue', ax=ax)
data_positive_loss[['Date', 'Exceedence ES 99%']].plot.scatter(
    x='Date', y='Exceedence ES 99%', color='black', ax=ax)
data_positive_loss[['Date', 'VaR 95%']].plot.line(
    x='Date', linestyle='-', color='red', ax=ax)
data_positive_loss[['Date', 'VaR 99%']].plot.line(
    x='Date', linestyle='--', color='red', ax=ax)
data_positive_loss[['Date', 'ES 95%']].plot.line(
    x='Date', linestyle='-', color='blue', ax=ax)
data_positive_loss[['Date', 'ES 99%']].plot.line(
    x='Date', linestyle='--', color='blue', ax=ax)
plot.show()
