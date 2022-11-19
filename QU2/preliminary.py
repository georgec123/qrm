import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plot

data = pd.read_csv('CW1-QU2-DATA')
date = data['Date']
close = data['DOG']
C = len(data)

ret = [(close[i] - close[i - 1])/close[i - 1] for i in range(1, C)]
ret.insert(0, 0)
log_ret = [(math.log(close[i]) - math.log(close[i - 1])) for i in range(1, C)]
log_ret.insert(0, 0)
loss_ret = [-x for x in log_ret]
data['Returns'] = ret
data['Log Returns'] = log_ret
data['Loss'] = loss_ret
