# Version created: 31 October 2018
import numpy as np
import pandas as pd
from sklearn import linear_model

import Reg_Oracle_Class

def get_group(A, X, X_sens, y_g, FP):
    """Given decisions on X, sensitive attributes, labels, and FP rate audit wrt
        to gamma unfairness. Return the group found, the gamma unfairness, fp disparity, and sign(fp disparity).
    """

    A_0 = [a for u, a in enumerate(A) if y_g[u] == 0]
    X_0 = pd.DataFrame([X_sens.iloc[u, :]
                        for u, s in enumerate(y_g) if s == 0])
    m = len(A_0)
    n = float(len(y_g))
    cost_0 = [0.0] * m
    cost_1 = -1.0 / n * (FP - A_0)
    reg0 = linear_model.LinearRegression()
    reg0.fit(X_0, cost_0)
    reg1 = linear_model.LinearRegression()
    reg1.fit(X_0, cost_1)
    func = Reg_Oracle_Class.RegOracle(reg0, reg1)
    group_members_0 = func.predict(X_0)
    err_group = np.mean([np.abs(group_members_0[i] - A_0[i])
                         for i in range(len(A_0))])
    # get the false positive rate in group
    if sum(group_members_0) == 0:
        fp_group_rate = 0
    else:
        fp_group_rate = np.mean([r for t, r in enumerate(A_0) if group_members_0[t] == 1])
    g_size_0 = np.sum(group_members_0) * 1.0 / n
    fp_disp = np.abs(fp_group_rate - FP)
    fp_disp_w = fp_disp * g_size_0

    # negation
    cost_0_neg = [0.0] * m
    cost_1_neg = -1.0 / n * (A_0-FP)
    reg0_neg = linear_model.LinearRegression()
    reg0_neg.fit(X_0, cost_0_neg)
    reg1_neg = linear_model.LinearRegression()
    reg1_neg.fit(X_0, cost_1_neg)
    func_neg = Reg_Oracle_Class.RegOracle(reg0_neg, reg1_neg)
    group_members_0_neg = func_neg.predict(X_0)
    err_group_neg = np.mean(
        [np.abs(group_members_0_neg[i] - A_0[i]) for i in range(len(A_0))])
    if sum(group_members_0_neg) == 0:
        fp_group_rate_neg = 0
    else:
        fp_group_rate_neg = np.mean([r for t, r in enumerate(A_0) if group_members_0[t] == 0])
    g_size_0_neg = np.sum(group_members_0_neg) * 1.0 / n
    fp_disp_neg = np.abs(fp_group_rate_neg - FP)
    fp_disp_w_neg = fp_disp_neg*g_size_0_neg

    # return group
    if fp_disp_w_neg > fp_disp_w:
        return [func_neg, fp_disp_w_neg, fp_disp_neg, err_group_neg, -1]
    else:
        return [func, fp_disp_w, fp_disp, err_group, 1]


def audit(predictions, X, X_prime, y):
    """Takes in predictions on dataset (X, X',y) and prints gamma-unfairness,
    fp disparity, group size, group coefficients, and sensitive column names.
    """
    FP = np.mean([p for i,p in enumerate(predictions) if y[i] == 0])
    aud_group, gamma_unfair, fp_in_group, err_group, pos_neg = get_group(predictions, X_sens=X_prime, X=X, y_g=y, FP=FP)

    return gamma_unfair