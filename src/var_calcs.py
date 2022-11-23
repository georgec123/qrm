from arch import arch_model
import numpy as np


def es(loss: np.ndarray, var: np.ndarray):
    """
    Calculate expected shortfall.
    Takes array of length n+1, and calculates avg(loss) of first n items given loss[0:n]>var[-1]

    :param loss: array of n+1 observations
    :param var: array of n+1 observations, but we only care about the 'final' element 
    """
    # check if loss[0 to n inclusive] > var[n+1]
    loss = loss[:-1]
    breach_mask = loss>var[-1]
    if not breach_mask.sum():
        return 0
    return loss[breach_mask].sum() / breach_mask.sum()

def es_n(loss: np.ndarray, var: float):
    """
    Calculate expected shortfall.
    Takes array of length n, and calculates avg(loss) given loss[0:n]>var

    :param loss: array of n observations, previous loss 
    :param var: Value at Risk
    """
    # check if loss[0 to n inclusive] > var
    breach_mask = loss>var
    if not breach_mask.sum():
        # return = for no breaches
        return 0
    return loss[breach_mask].sum() / breach_mask.sum()



def garch_var(x, alpha):
    
    model = arch_model(x.dropna(),
                    mean='constant', 
                    vol='GARCH', 
                    p=1, q=1, rescale=True, dist='normal')

    model_fit = model.fit(update_freq=-1, disp=0)
    forecasts = model_fit.forecast(reindex=False)
    var = forecasts.variance.iloc[0,0]
    q = np.quantile(model_fit.std_resid.values, alpha)
    std = np.sqrt(var)
    value_at_risk = q*std + model_fit.params.mu
    return value_at_risk

def garch_es(x, alpha):
        
    model = arch_model(x.dropna(),
                    mean='constant', 
                    vol='GARCH', 
                    p=1, q=1, rescale=True, dist='normal')

    model_fit = model.fit(update_freq=-1, disp=0)
    forecasts = model_fit.forecast(reindex=False)
    var = forecasts.variance.iloc[0,0]
    q = np.quantile(model_fit.std_resid, alpha)
    res_exp_sh = es_n(model_fit.std_resid, q)
    
    exp_sh = (var**0.5)*res_exp_sh
    return exp_sh