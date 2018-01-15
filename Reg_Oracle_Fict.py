# take CL arguments B, num_sens, printflag, dataset, oracle, max_iters, beta
# B: max dual norm
# beta: disparity tolerance
# printflag: bool determines whether things are printed at each iteration
# or only at the end
# num_sens: number of sensitive features, in 1:18
# oracle: 'reg_oracle'
# dataset: name of the dataset to use

# run from command line: python Reg_Oracle_Fict.py 50 17 True communities
# reg_oracle 10000 .001

import sys
# get command line arguments
B, num_sens, printflag, dataset, oracle, max_iters, beta = sys.argv[1:]
num_sens = int(num_sens)
printflag = sys.argv[3].lower() == 'true'
B = float(B)
dataset = str(dataset)
oracle = str(oracle)
max_iters = int(max_iters)
beta = float(beta)

import clean_data
import numpy as np
import pandas as pd
from sklearn import linear_model
import random
import Reg_Oracle_Class
random.seed(1)

# print out the invoked parameters
print(
    'Invoked Parameters: C = {}, number of sensitive attributes = {}, random seed = 1,dataset = {}, learning oracle = {}'.format(
        B,
        num_sens,
        dataset,
        oracle))


# Data Cleaning and Import
f_name = 'clean_{}'.format(dataset)
clean_the_dataset = getattr(clean_data, f_name)
X, X_prime, y = clean_the_dataset(num_sens)

# -----------------------------------------------------------------------------------------------------------

# Heuristic Frank-Wolfe Algorithm for Finding Optimally (FP)-Fair Classifiers

# Inputs:
# data set: (x,x',y), y in [0,1]
# x' are protected features
# A: Oracle for agnostic learning Y via (x,x')
# F: group set (some concept class over x')
# Aud: Oracle for learning A via F | Y = 0
# C parameter controlling the norm of dual variables
# alpha: unfairness tolerance parameter.

# Algorithm
# Initialize p_0 to be the distribution minimizing empirical error A(X,y)
# At time t, given classifiers selected in rounds 1...t-1, sample one uniformly and classify (x,x') | y = 0
# Run Aud on A | (x,x',0) -> this produces a group j*. Check FP(j)-FP. If < 0 set C = -C
# if FP(j)-FP < alpha, stop, output {p_t-1}.
# Re-weight examples as follows:
# y = 0, x' in j*, A = 1 -> w = 1 + C(1/n_{0j}-1/n_0)
# y = 0 x' not in j* -> w = 1-C/n_0
# all other points: w = 1
# let p_t <- A(Y, X, w)
# -----------------------------------------------------------------------------------------------------------
# Helper Functions
# given a sequence of classifiers p, returns decisions on Data set X

# Inputs:
# A: the previous set of decisions (probabilities) up to time iter - 1
# q: the most recent classifier found
# x: the dataset
# y: the labels
# iter: the iteration
#
# Outputs:
# error: the error of the average classifier found thus far
#


def gen_a(q, x, y, A, iter):
    new_preds = np.multiply(1.0 / iter, q.predict(x))
    ds = np.multiply((iter - 1.0) / iter, A)
    ds = np.add(ds, new_preds)
    error = np.mean([np.abs(ds[k] - y[k]) for k in range(len(y))])
    return [error, ds]


# given an algorithms decisions empirical history of decisions A, sensitive variables X_sense, and true y values y_g
# returns the best classifier learning A via X_sense on the set y_g = 0
# K: number of draws from p where we take the subgroup with largest
# discrimination
def get_group(A, p, X, X_sens, y_g, FP, beta):

    A_0 = [a for u, a in enumerate(A) if y_g[u] == 0]
    X_0 = pd.DataFrame([X_sens.iloc[u, :]
                        for u, s in enumerate(y_g) if s == 0])
    m = len(A_0)
    cost_0 = [0] * m
    cost_1 = -1 / m * ((FP - beta) - A_0)
    reg0 = linear_model.LinearRegression()
    reg0.fit(X_0, cost_0)
    reg1 = linear_model.LinearRegression()
    reg1.fit(X_0, cost_1)
    func = Reg_Oracle_Class.RegOracle(reg0, reg1)
    group_members_0 = func.predict(X_0)
    err_group = np.mean([np.abs(group_members_0[i] - A_0[i])
                         for i in range(len(A_0))])
    # get the false positive rate in group
    fp_group_rate = np.mean(
        [r for t, r in enumerate(A_0) if group_members_0[t] == 1])
    fp_disp_rate = np.abs(fp_group_rate - FP)

    # negation
    cost_0_neg = [0] * m
    cost_1_neg = -1 / m * (A_0 - (FP + beta))
    reg0_neg = linear_model.LinearRegression()
    reg0_neg.fit(X_0, cost_0_neg)
    reg1_neg = linear_model.LinearRegression()
    reg1_neg.fit(X_0, cost_1_neg)
    func_neg = Reg_Oracle_Class.RegOracle(reg0, reg1)
    group_members_0_neg = func_neg.predict(X_0)
    err_group_neg = np.mean(
        [np.abs(group_members_0_neg[i] - A_0[i]) for i in range(len(A_0))])
    fp_group_rate_neg = np.mean(
        [r for t, r in enumerate(A_0) if group_members_0[t] == 0])
    fp_disp_rate_neg = np.abs(fp_group_rate_neg - FP)

    # return group
    if fp_disp_rate_neg > fp_disp_rate:
        return [func_neg, fp_disp_rate_neg, fp_group_rate_neg, err_group_neg]
    else:
        return [func, fp_disp_rate, fp_group_rate, err_group]


# p is a classifier
# X is the data
# X_sens is the sensitive data
# y_g are the values
# g is the group
# calculates the false positive rate disparity of p with respect to a
# specific group g
def calc_disp(p, X, y_g, X_sens, g):
    A_p = p.predict(X)
    FP = [A_p[i] for i, c in enumerate(y_g) if c == 0]
    FP = np.mean(FP)
    group_members = g.predict(X_sens)
    fp_g = [A_p[i]
            for i, c in enumerate(y_g) if group_members[i] == 1 and c == 0]
    if len(fp_g) == 0:
        return 0
    fp_g = np.mean(fp_g)
    return np.abs(FP - fp_g)


# given a sequence of classifiers we want to print out the unfairness in
# each marginal coordinate
def calc_unfairness(A, X_prime, y_g, FP_p):
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


# update c1 for y = 0
def learner_costs(c_1, f, X_prime, y, B, iteration):
    if iteration == 1:
        return c_1
    fp_g = f[2]
    X_0_prime = pd.DataFrame([X_prime.iloc[u, :]
                              for u, s in enumerate(y) if s == 0])
    g_members = f[0].predict(X_0_prime)
    m = len(c_1)
    for t in range(m):
        c_1[t] = (c_1[t] - 1.0 / m) * (iteration / (iteration - 1.0)) + \
            B / iteration * g_members[t] * (fp_g - 1) + 1.0 / m
    return c_1


def learner_br(c_1t, X, y):
    c_1t_new = c_1t[:]
    n = len(y)
    c_0 = [0] * n
    c_1 = []
    for s in range(n):
        if y[s] == 0:
            c_1.append(-1.0 / n)
        else:
            c_1.append(c_1t_new.pop(0))
    reg0 = linear_model.LinearRegression()
    reg0.fit(X, c_0)
    reg1 = linear_model.LinearRegression()
    reg1.fit(X, c_1)
    func = Reg_Oracle_Class.RegOracle(reg0, reg1)
    return func


# -----------------------------------------------------------------------------------------------------------
# Fictitious Play Algorithm

stop = False
n = X.shape[0]
# initialize classifier with random weighting of the dataset
# w initial weighting
p = [fit_weighted([1.0 / n] * n, X, y, oracle)[0]]
iteration = 1
errors_t = []
fp_diff_t = []
coef_t = []
size_t = []
groups = []
cum_group_mems = []
# correspond to dual player first playing the group that is everyone
#w = [0.0] * n
m = len([s for s in y if s == 0])
c_1t = [1.0 / m] * m
FP = 0
A = [0.0] * n
group_membership = [0.0] * n
while iteration < max_iters:
    print('iteration: {}'.format(iteration))
    # get algorithm decisions on X by randomizing on current set of p
    emp_p = gen_a(p[-1], X, y, A, iteration)
    # get the error of the classifier
    err = emp_p[0]
    # Average decisions
    A = emp_p[1]
    # get the false positive rate of the classifier overall
    A_sample = p[-1].predict(X)
    FP_new = np.mean([A_sample[i] for i, c in enumerate(y) if c == 0])
    FP = ((iteration - 1.0) / iteration) * FP + FP_new * (1.0 / iteration)
    # dual player best responds: audit A via F, to get a group f: best
    # response to strategy up to t-1
    f = get_group(A, p, X, X_prime, y, FP, beta)
    group_membership = np.add(group_membership, f[0].predict(X_prime))
    group_membership = [g != 0 for g in group_membership]
    # cumulative group members up to time t
    group_members_t = np.sum(group_membership)
    cum_group_mems.append(group_members_t)

    # primal player best responds: cost-sensitive classification
    p_t = learner_br(c_1t, X, y)

    # calculate the FP rate of the new p_t on the last group found
    fp_rate_after_fit = 0
    if iteration > 1:
        fp_rate_after_fit = calc_disp(
            p_t, X, y, X_prime, groups[len(groups) - 1])
    # append new group, new p, fp_diff of group found, coefficients, group size
    groups.append(f[0])
    p.append(p_t)
    fp_diff_t.append(np.abs(f[1]))
    errors_t.append(err)
    coef_t.append(f[0].b0.coef_ - f[0].b1.coef_)
    group_train = f[0].predict(X_prime)
    size_t.append(np.mean(group_train))
    if iteration == 1:
        print(
            'most accurate classifier accuracy: {}, most acc-class unfairness: {}, most acc-class size {}'.format(
                err,
                fp_diff_t[0],
                size_t[0]))
    # get unfairness on marginal subgroups
    unfairness = calc_unfairness(A, X_prime, y, FP)
    # print
    if printflag:
        print('XX av error time {}, FP group diff, Group Size, Err Audit, FP Rate Diff Lag, Lgrgian err p_t, Cum_group: {} {} {} {} {} {}'.format(iteration, '{:f}'.format(
            err), '{:f}'.format(np.abs(f[1])), '{:f}'.format(np.mean(group_train)), '{:f}'.format(f[3]), '{:f}'.format(fp_rate_after_fit), '{:f}'.format(cum_group_mems[-1])))
        group_coef = f[0].b0.coef_ - f[0].b1.coef_
        print('YYY coefficients of g_t: {}'.format(group_coef),)
        print('Unfairness in marginal subgroups: {}'.format(unfairness),)

    # update costs: dual player best responds
    c_1t = learner_costs(c_1t, f, X_prime, y, B, iteration)

    sys.stdout.flush()
    iteration += 1


# evaluate fair classifier found
D = gen_a(p[-1], X, y, A, iteration)
error_D = D[0]
model = fit_weighted([1.0 / n] * n, X, y, oracle)[0]
preds = model.predict(X)
error_opt = np.mean([np.abs(c - preds[i]) for i, c in enumerate(y)])
best_A = gen_a(model, X, y, [0.0] * n, iteration)
print('\n')
print('final classifier error on the data set: {}'.format(error_D))
print('best classifier error on the data set: {}'.format(error_opt))
print(
    'best classifier unfairness on the data set: {}'.format(
        fp_diff_t[0]))
print('FP base rate difference over time: {}'.format(fp_diff_t))
print('Classifier error over time: {}'.format(errors_t))
print('Group Size over time: {}'.format(size_t))
