import pandas as pd
import numpy as np
from scipy import stats
#this is a comment

def log_lik(p: float, obs: pd.Series):
    """
    Likelihood ratio = -2*log(L_1/L_2)
    This function will calculate log(L_i) for any series of binomial obesrvations

    :param p: probability of success
    :param obs: Series of observations, values are 1 or 0
    """
    return obs.apply(lambda x: np.log(((1-p)**(1-x))*(p**x))).sum()

def lr_ind(alpha, exceeds: pd.Series):
    p_h0 = 1-alpha
    # fill value doesnt matter as we slice out. just need bool to retain dtype
    # look at obervations from t2 to tn
    obs_1x = exceeds[1:][exceeds.shift(1, fill_value=False)[1:]]
    obs_0x = exceeds[1:][~exceeds.shift(1, fill_value=False)[1:]]

    t_11 = obs_1x.sum()
    t_10 = len(obs_1x) - t_11

    t_01 = obs_0x.sum()
    t_00 = len(obs_0x) - t_01

    
    pi_11 = obs_1x.mean()
    pi_01 = obs_0x.mean()

    # print(pi_11, pi_01)
    ll_d = t_00*np.log(1-pi_01) + t_01*np.log(pi_01) + t_10*np.log(1-pi_11) 
    if pi_11:
        # account for pi_11 being 0
        ll_d+=  t_11*np.log(pi_11)
    ll_n = t_00*np.log(1-p_h0) + t_01*np.log(p_h0) + t_10*np.log(1-p_h0) + t_11*np.log(p_h0)

    return -2*(ll_n-ll_d)


def lr_uc(alpha: float, exceeds: pd.Series):
    """
    Calculate unconditional likelihood ratio statistic for any series of binomial obesrvations

    :param alpha: confidence level between 0 and 1
    :param exceeds: Series of booleans/ints, indicating if the event happened
    """
    # calc pi_hat MLE 
    obs = exceeds.astype(int)
    pi_hat = np.mean(obs)

    # return LR_UC for pi_hat and 1-alpha
    return -2*(log_lik(1-alpha, obs)-log_lik(pi_hat, obs))

def p_chi2(x: float, dof: int=1):
    """
    Calculate the p value for a chi2 distrubution

    :param x: test statistic value
    :param dof: Degrees of freedom for chi2 ist
    """
    return 1-stats.chi2.cdf(x, dof)

def str_ci(x):
    return str(x).split('.')[1]

def get_stats(data, title: str): 
    df=pd.DataFrame(columns=['Alpha', 'Violations (exp)', 'LR_uc', 'LR_uc - p','LR_joint', 'LR_joint - p', 'ES - p'])

    for alpha in [0.95, 0.99]:

        # str alpha for dataframe column lookup
        str_alpha = str_ci(alpha)

        # remove start data where there is no var
        non_na_data = data[~data[f'var_{str_alpha}'].isna()]

        num_days = len(non_na_data)
        viol_mask = (non_na_data['loss']>non_na_data[f'var_{str_alpha}'])
        num_viols = viol_mask.sum()


        # calculate p values for var
        expected_viols = (1-alpha)*num_days

        likelihood_uc = lr_uc(alpha, viol_mask)
        likelihood_ind = lr_ind (alpha, viol_mask)

        p_val_uc = p_chi2(likelihood_uc)
        p_val_joint = p_chi2(likelihood_uc+likelihood_ind, dof=2)

        accept_str = lambda p_val, alpha : f"{'Accept' if p_val>(1-alpha) else 'Reject'}"

        accept_uc = accept_str(p_val_uc, alpha)
        accept_joint = accept_str(p_val_joint, alpha)

        # calculate p values for ES
        viols = non_na_data[viol_mask]
        xis = viols['loss']-viols[f'es_{str_alpha}']
        z = xis.sum()/np.sqrt((xis**2).sum())
        p_es = 1-stats.norm.cdf(z)

        # print(z, p_es)
        accept_es = accept_str(p_es, alpha)

    
        # print(f"\n{alpha=}. Violations (exp): {num_viols} ({expected_viols:.2f}).")
        # print(f"VaR LR_uc= {likelihood_uc:.3f}. p-val: {p_val_uc:.5f}. {accept_uc}")                
        # print(f"VaR LR_joint= {likelihood_uc+likelihood_ind:.3f}. p-val: {p_val_joint:.5f}. {accept_joint}")                
        # print(f"ES: Z={z:.2f}. p-val: {p_es:.5f}. {accept_es}")

        df.loc[len(df),:] = [alpha, f"{num_viols} ({expected_viols:.1f})", f"{likelihood_uc:.3f}",
            f"{p_val_uc:.5f}", f"{likelihood_uc+likelihood_ind:.3f}", f"{p_val_joint:.5f}",
            f"{p_es:.5f}" ]
    df.insert(0, 'Title', title)
    return df
