def L(pi, T, num_violations):
    not_violation = (1 - pi) * (T - num_violations)
    violation = pi * num_violations

    return not_violation * violation

def MLE(T, num_violations):
    res = (1/T) * num_violations

    return res

def LR_UC(alpha, T, num_violations):
    numerator = L(1 - alpha, T, num_violations)
    test_stat = MLE(T, num_violations)
    denominator = L(test_stat, T, num_violations)

    res = -2 * math.log(numerator/denominator)

    return res
  
 def p_val(alpha, T, num_violations):
    LR = LR_UC(alpha, T, num_violations)
    p = (1 - stats.chi2.cdf(LR, 1))

    return p
