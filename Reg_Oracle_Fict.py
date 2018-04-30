# Version Created: 20 April 2018
# import matplotlib
# matplotlib.use('TkAgg')
import clean_data
import numpy as np
import pandas as pd
from sklearn import linear_model
import random
import Reg_Oracle_Class
import sys
# from matplotlib import pyplot as plt
import heatmap
from MSR_Reduction import *

# USAGE: python Reg_Oracle_Fict.py 100 True communities reg_oracle 5 .006 'gamma'


# Helper Functions
# -----------------------------------------------------------------------------------------------------------

# Inputs:
# A: the previous set of decisions (probabilities) up to time iter - 1
# q: the most recent classifier found
# x: the dataset
# y: the labels
# iter: the iteration
# Outputs:
# error: the error of the average classifier found thus far (incorporating q)
def gen_a(q, x, y, A, iter):
    """Return the classifications of the average classifier at time iter."""

    new_preds = np.multiply(1.0 / iter, q.predict(x))
    ds = np.multiply((iter - 1.0) / iter, A)
    ds = np.add(ds, new_preds)
    error = np.mean([np.abs(ds[k] - y[k]) for k in range(len(y))])
    return [error, ds]


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


def learner_costs(c_1, f, X_prime, y, C, iteration, fp_disp, gamma):
    """Recursively update the costs from incorrectly predicting 1 for the learner."""
    # store whether FP disparity was + or -
    pos_neg = f[4]
    X_0_prime = pd.DataFrame([X_prime.iloc[u, :] for u,s in enumerate(y) if s == 0])
    g_members = f[0].predict(X_0_prime)
    m = len(c_1)
    n = float(len(y))
    g_weight_0 = np.sum(g_members)*(1.0/float(m))
    for t in range(m):
        new_group_cost = (1.0/n)*pos_neg*C*(1.0/iteration) * g_members[t] * (g_weight_0 - 1)
        if np.abs(fp_disp) < gamma:
            if t == 0:
                print('barrier')
            new_group_cost = 0
        c_1[t] = (c_1[t] - 1.0/n) * ((iteration-1.0)/iteration) + new_group_cost + 1.0/n
    return c_1


def learner_br(c_1t, X, y):
    """Solve the CSC problem for the learner."""
    n = len(y)
    c_1t_new = c_1t[:]
    c_0 = [0.0] * n
    c_1 = []
    for r in range(n):
        if y[r] == 1:
            c_1.append((-1.0/n))
        else:
            c_1.append(c_1t_new.pop(0))
    reg0 = linear_model.LinearRegression()
    reg0.fit(X, c_0)
    reg1 = linear_model.LinearRegression()
    reg1.fit(X, c_1)
    func = Reg_Oracle_Class.RegOracle(reg0, reg1)
    return func


# not currently printed
def lagrangian_value(groups, yhat, C, FP, X, X_prime, y, iteration):
    """Compute the lagrangian value wrt to a learner yhat and found groups in groups."""
    lagrange = 0
    n = len(y)
    err_pt = np.mean([np.abs(yhat[r]-y[r]) for r in range(n)])
    for g in groups:
        g_mems = g.predict(X_prime)
        fp_g = np.mean([yhat[i] for i in range(n) if y[i] == 0 and g_mems[i] == 1])
        fp_disp = fp_g-FP
        group_size_0 = np.sum(f[0].predict(X_0)) * (1.0/n)
        lagrange += C*1.0/(iteration-1.0)*fp_disp*group_size_0
    return lagrange + err_pt

# not currently printed out
def calc_unfairness(A, X_prime, y_g, FP_p):
    """Calculate unfairness in marginal sensitive subgroups (thresholded)."""
    unfairness = []
    n = X_prime.shape[1]
    sens_means = np.mean(X_prime, 0)
    for q in range(n):
        group_members = [X_prime.iloc[i, q] > sens_means[q]
                         for i in range(X_prime.shape[0])]
        # calculate FP rate on group members
        fp_g = [a for t, a in enumerate(
            A) if group_members[t] == 1 and y_g[t] == 0]
        if len(fp_g) > 0:
            fp_g = np.mean(fp_g)
        else:
            fp_g = 0
        # calculate the fp rate on non-group members
        group_members_neg = [1 - g for g in group_members]
        # calculate FP rate on group members
        fp_g_neg = [a for t, a in enumerate(
            A) if group_members_neg[t] == 1 and y_g[t] == 0]
        if len(fp_g_neg) > 0:
            fp_g_neg = np.mean(fp_g_neg)
        else:
            fp_g_neg = 0
        unfairness.append(
            np.max([np.abs(np.mean(fp_g) - FP_p), np.abs(np.mean(fp_g_neg) - FP_p)]))
    return unfairness


# -----------------------------------------------------------------------------------------------------------
# Fictitious Play Algorithm

if __name__ == "__main__":

    # get command line arguments
    # C, num_sens, printflag, dataset, oracle, max_iters, gamma, fairness_def, num, col = 100, 2, True, 'communities', 'reg_oracle', 10, .0001, 'gamma', 100, 18
    C, printflag, dataset, oracle, max_iters, gamma, fairness_def = sys.argv[1:]
    printflag = sys.argv[2].lower() == 'true'
    C = float(C)
    dataset = str(dataset)
    oracle = str(oracle)
    max_iters = int(max_iters)
    gamma = float(gamma)
    fairness_def = str(fairness_def)
    random.seed(1)

    # Data Cleaning and Import
    f_name = 'clean_{}'.format(dataset)
    clean_the_dataset = getattr(clean_data, f_name)

    X, X_prime, y = clean_the_dataset()


    # print out the invoked parameters
    print('Invoked Parameters: C = {}, number of sensitive attributes = {}, random seed = 1, dataset = {}, learning oracle = {}, gamma = {}, formulation: {}'.format(C, X_prime.shape[1], dataset, oracle, gamma, fairness_def))

    # subsample
    # if num > 0:
    #     X = X.iloc[0:num, 0:col]
    #     y = y[0:num]
    #     X_prime = X_prime.iloc[0:num, :]

    stop = False
    n = X.shape[0]
    m = len([s for s in y if s == 0])
    p = [learner_br([1.0/n]*m, X, y)]
    iteration = 1
    errors_t = []
    fp_diff_t = []
    coef_t = []
    size_t = []
    groups = []
    cum_group_mems = []
    m = len([s for s in y if s == 0])
    c_1t = [1.0 / n] * m
    FP = 0
    A = [0.0] * n
    group_membership = [0.0] * n
    X_0 = pd.DataFrame([X_prime.iloc[u, :] for u, s in enumerate(y) if s == 0])

    while iteration < max_iters:
        print('iteration: {}'.format(iteration))
        # get t-1 mixture decisions on X by randomizing on current set of p
        emp_p = gen_a(p[-1], X, y, A, iteration)
        # get the error of the t-1 mixture classifier
        err = emp_p[0]
        # Average decisions
        A = emp_p[1]
        # store intermediate A for heatmap
        if iteration == max_iters/5:
            A_1 = A

        if iteration == 2*max_iters/5:
            A_2 = A

        if iteration == 3*max_iters/5:
            A_3 = A

        if iteration == 4*max_iters/5:
            A_4 = A

        # update FP to get the false positive rate of the mixture classifier
        A_recent = p[-1].predict(X)
        # FP rate of t-1 mixture on new group g_t
        FP_recent = np.mean([A_recent[i] for i, c in enumerate(y) if c == 0])
        FP = ((iteration - 1.0) / iteration) * FP + FP_recent * (1.0 / iteration)
        # dual player best responds to strategy up to t-1
        f = get_group(A, X, X_prime, y, FP)
        # flag whether FP disparity was positive or negative
        pos_neg = f[4]
        fp_disparity = f[1]
        group_size_0 = np.sum(f[0].predict(X_0)) * (1.0 / n)

        # compute list of people who have been included in an identified subgroup up to time t
        group_membership = np.add(group_membership, f[0].predict(X_prime))
        group_membership = [g != 0 for g in group_membership]
        group_members_t = np.sum(group_membership)
        cum_group_mems.append(group_members_t)

        # primal player best responds: cost-sensitive classification
        p_t = learner_br(c_1t, X, y)
        A_t = p_t.predict(X)
        FP_t = np.mean([A_t[i] for i, c in enumerate(y) if c == 0])
        # get lagrangian value which primal player is minimizing
        # lagrange = lagrangian_value(groups, A_t, C, FP_t, X, X_prime, y, iteration)

        # append new group, new p, fp_diff of group found, coefficients, group size
        groups.append(f[0])
        p.append(p_t)
        fp_diff_t.append(np.abs(f[1]))
        errors_t.append(err)
        coef_t.append(f[0].b0.coef_ - f[0].b1.coef_)

        if iteration == 1:
            print(
                'most accurate classifier accuracy: {}, most acc-class unfairness: {}, most acc-class size {}'.format(
                    err,
                    fp_diff_t[0],
                    group_size_0))
        # get unfairness on marginal subgroups
        # unfairness = calc_unfairness(A, X_prime, y, FP)
        # print
        if printflag:
            print('ave error: {}, gamma-unfairness: {}, group_size: {}, frac included ppl: {}'.format('{:f}'.format(err), '{:f}'.format(np.abs(f[1])), '{:f}'.format(group_size_0), '{:f}'.format(cum_group_mems[-1]/float(n))))
            group_coef = f[0].b0.coef_ - f[0].b1.coef_
            print('YY coefficients of g_t: {}'.format(list(group_coef)))

        # update costs: the primal player best responds
        c_1t = learner_costs(c_1t, f, X_prime, y, C, iteration, fp_disparity, gamma)
        # print('UU learner costs: {}'.format(np.unique(c_1t)))
        sys.stdout.flush()
        iteration += 1
        iteration = float(iteration)

    # plot errors
    # x = range(max_iters-1)
    # y_t = errors_t
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax1.plot(x,y_t)
    # plt.ylabel('average error of mixture')
    # plt.xlabel('iterations')
    # plt.title('error vs. time: C: {}, gamma: {}, dataset: {}'.format(C, gamma, dataset))
    # ax1.plot(x, [np.mean(y_t)]*len(y_t))
    # plt.clf()
    #
    # # plot fp disparity
    # x = range(max_iters-1)
    # y_t = fp_diff_t
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # ax2.plot(x, y_t)
    # plt.ylabel('fp_diff*group_size')
    # plt.xlabel('iterations')
    # plt.title('fp_diff*size vs. time: C: {}, gamma: {}, dataset: {}'.format(C, gamma, dataset))
    # ax2.plot(x, [gamma]*len(y_t))
    # plt.clf()

    # initial heat map
    X_prime = X_prime.iloc[:, 0:2]
    eta = .05

    minimax1 = heatmap.heat_map(X, X_prime, y, p[0].predict(X), eta, 'starting', None, None)

    minimax2 = heatmap.heat_map(X, X_prime, y, A_2, eta, 'intermediate1', mini=minimax1[0], maxi=minimax1[1])

    minimax3 = heatmap.heat_map(X, X_prime, y, A_3, eta, 'intermediate2', mini=minimax1[0], maxi=minimax1[1])

    minimax4 = heatmap.heat_map(X, X_prime, y, A_4, eta, 'intermediate3', mini=minimax1[0], maxi=minimax1[1])

    minimax5 = heatmap.heat_map(X, X_prime, y, A, eta, 'ending', mini=minimax1[0], maxi=minimax1[1])

    # MSR heat map
    X_prime_cts = X_prime.copy()
    # threshold sensitive features by average value
    sens_means = np.mean(X_prime)
    for col in X_prime.columns:
        X.loc[(X[col] > sens_means[col]), col] = 1
        X_prime.loc[(X_prime[col] > sens_means[col]), col] = 1
        X.loc[(X[col] <= sens_means[col]), col] = 0
        X_prime.loc[(X_prime[col] <= sens_means[col]), col] = 0
    A_MSR = MSR_preds(X, X_prime, X_prime_cts, y, max_iters, False)

    minimax4 = heatmap.heat_map(X, X_prime_cts, y, A_MSR, eta, 'MSR_ending', mini=minimax1[0], maxi=minimax1[1])


