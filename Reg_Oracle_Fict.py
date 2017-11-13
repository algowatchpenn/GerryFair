# take CL arguments B, alpha, printflag, K, plotflag, dataset
# B: max dual norm
# alpha: disparity tolerance
# printflag: bool determines whether things are printed at each iteration
# or only at the end
# num_sens: number of sensitive features, in 1:18
# K: number of times we draw a random classification of the dataset and audit at each round
# plotflag: bool indicating whether to generate plots
# dataset: name of the dataset to use

import sys
# get command line arguments
B, num_sens, printflag, K, plotflag, dataset, oracle, aud_oracle, max_iters = sys.argv[1:]
num_sens = int(num_sens)
printflag = sys.argv[3].lower() == 'true'
B = float(B)
plotflag = sys.argv[5].lower() == 'true'
K = int(K)
dataset = str(dataset)
oracle = str(oracle)
aud_orc = str(aud_oracle)
max_iters = int(max_iters)
#B, num_sens, printflag, K, plotflag, dataset, oracle, aud_orc, max_iters = 6,18,True,2,False,'communities','reg_oracle','reg_oracle',100
import clean_data
import numpy as np
import pandas as pd
from sklearn import linear_model
#from matplotlib import pyplot as plt
import random
from sklearn import svm
import Reg_Oracle_Class
from sklearn import ensemble
from sklearn import dummy
random.seed(1)

# print out the invoked parameters
print(
    'Invoked Parameters: C = {}, number of sensitive attributes = {}, K = {}, random seed = 1, '
    'plots = {}, data set = {}, learning oracle = {}, auditing oracle = {}'.format(
        B,
        num_sens,
        K,
        plotflag,
        dataset,
        oracle, aud_orc))


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
    new_preds = np.multiply(1.0/iter, q.predict(x))
    ds = np.multiply((iter-1.0)/iter, A)
    ds = np.add(ds, new_preds)
    error = np.mean([np.abs(ds[k]-y[k]) for k in range(len(y))])
    return [error, ds]

# given an algorithms decisions A, sensitive variables X_sense, and true y values y_g
# returns the best classifier learning A via X_sense on the set y_g = 0
# K: number of draws from p where we take the subgroup with largest
# discrimination
def get_group(A, p, X, X_sens, y_g, K, aud_orc, FP):

    funcs = []
    fp_rates = []
    fp_disp = []
    errs = []

    # regression oracle case
    if aud_orc == 'reg_oracle':
        A_0 = [a for u,a in enumerate(A) if y_g[u] == 0]
        X_0 = pd.DataFrame([X_sens.iloc[u, :] for u, s in enumerate(y_g) if s == 0])
        cost_0 = A_0[:]
        cost_1 = [1-t for t in A_0]
        reg0 = linear_model.LinearRegression()
        reg0.fit(X_0, cost_0)
        reg1 = linear_model.LinearRegression()
        reg1.fit(X_0, cost_1)
        func = Reg_Oracle_Class.RegOracle(reg0, reg1)
        group_members_0 = func.predict(X_0)
        err_group = np.mean([np.abs(group_members_0[i]-A_0[i]) for i in range(len(A_0))])
        # get the false positive rate in group
        fp_group_rate = np.mean([r for t, r in enumerate(A_0) if group_members_0[t] == 1])
        fp_disp_rate = np.abs(fp_group_rate - FP)

        # negation
        func_neg = Reg_Oracle_Class.RegOracle(reg1, reg0)
        group_members_0_neg = func_neg.predict(X_0)
        err_group_neg = np.mean([np.abs(group_members_0_neg[i]-A_0[i]) for i in range(len(A_0))])
        fp_group_rate_neg = np.mean([r for t, r in enumerate(A_0) if group_members_0[t] == 0])
        fp_disp_rate_neg = np.abs(fp_group_rate_neg - FP)
        if fp_disp_rate_neg > fp_disp_rate:
            return [func_neg, fp_disp_rate_neg, fp_group_rate_neg, err_group_neg]
        else:
            return [func, fp_disp_rate, fp_group_rate, err_group]

    # K = 0: blown up data set
    if K == 0:
        blowup_A = p[0].predict(X)
        y_blowup = y_g[:]
        blowup_ds = X_sens.iloc[:, :]
        for classifier in p[1:]:
            blowup_A += classifier.predict(X)
            y_blowup += y_g
            blowup_ds = pd.concat([blowup_ds, X_sens])

        clf1 = linear_model.LogisticRegression()
        # discard points where y = 1
        A_0 = [blowup_A[u] for u,s in enumerate(y_blowup) if s == 0]
        X_0 = pd.DataFrame([blowup_ds.iloc[u,:] for u,s in enumerate(y_blowup) if s == 0])
        func = clf1.fit(X_0, pd.DataFrame(A_0).values.ravel())
        # compute the false positive rates in the group
        group_members_0 = func.predict(X_0)
        err = np.mean([group_members_0[i] != A_0[i] for i in range(len(A_0))])
        fp_group_rate = np.mean([r for t, r in enumerate(A_0) if group_members_0[t] == 1])
        fp_disp_rate = np.abs(fp_group_rate - FP)

        # negation
        # compute the false positive rate in the group found
        fp_group_rate_neg = np.mean([r for t, r in enumerate(A_0) if group_members_0[t] == 0])
        fp_disp_rate_neg = np.abs(fp_group_rate_neg - FP)
        A_0_neg = [1-a for a in A_0]
        clf2 = linear_model.LogisticRegression()
        func_neg = clf2.fit(X_0,pd.DataFrame(A_0_neg).values.ravel())
        group_members_0_neg = func_neg.predict(X_0)
        err_neg = np.mean([group_members_0_neg[i] != A_0_neg[i] for i in range(len(A_0_neg))])
        if fp_disp_rate_neg > fp_disp_rate:
            return [func_neg, fp_disp_rate_neg, fp_group_rate_neg, err_neg]
        else:
            return [func, fp_disp_rate, fp_group_rate, err]
    # K > 0: sampling approach

    else:
        for i in range(K):
            # draw a sample dataset
            A_sample = [np.random.binomial(1, a, 1) for a in A]
            A_Y1 = [A_sample[k] for k, r in enumerate(y_g) if r == 0]
            X_Y1 = [X_sens.iloc[k, :] for k, r in enumerate(y_g) if r == 0]
            clf1 = linear_model.LogisticRegression()
            func = clf1.fit(X_Y1, pd.DataFrame(A_Y1).values.ravel())
            group_members_0 = func.predict(X_Y1)
            err_g = np.mean([np.abs(group_members_0[i] - A_Y1[i]) for i in range(len(A_Y1))])
            errs.append(err_g)
           # compute the false positive rates in the group
            fp_group_rate = np.mean([r for t, r in enumerate(A_Y1) if group_members_0[t] == 0])
            funcs.append(func)
            fp_rates.append(fp_group_rate)
            fp_disp_rate = np.abs(fp_group_rate - FP)
            fp_disp.append(fp_disp_rate)

            # get the group negation
            clf2 = linear_model.LogisticRegression()
            A_Y1_c = [np.abs(1 - a) for a in A_Y1]
            func_comp = clf2.fit(X_Y1, pd.DataFrame(A_Y1_c).values.ravel())
            group_members_0 = func_comp.predict(X_Y1)
            err_g_comp = np.mean([np.abs(group_members_0[i]-A_Y1_c[i]) for i in range(len(A_Y1_c))])
            errs.append(err_g_comp)
            # compute the false positive rate in the group found
            fp_group_rate = np.mean([r for t, r in enumerate(A_Y1_c) if group_members_0[t] == 0])
            fp_disp_rate = np.abs(fp_group_rate - FP)
            funcs.append(func_comp)
            fp_rates.append(fp_group_rate)
            fp_disp.append(fp_disp_rate)
            return [funcs[np.argmax(fp_disp)], fp_disp[np.argmax(fp_disp)], fp_rates[np.argmax(fp_disp)], errs[np.argmax(fp_disp)]]


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
    fp_g = [A_p[i] for i, c in enumerate(y_g) if group_members[i] == 1 and c == 0]
    if len(fp_g) == 0:
        return 0
    fp_g = np.mean(fp_g)
    return np.abs(FP - fp_g)


# fits a weighted classification oracle to the Data set (X,y_t) with
# weights w
def eval_weighted(q, x, y_t, classifier):
    score = 0
    y_preds = classifier.predict(x)
    score = 0
    for l, z in enumerate(y_t):
        if z != y_preds[l]:
            score += q[l]
    return score


def fit_weighted(q, x, y_t, orc):
    if orc == 'reg_oracle':
        cost_0 = [0 if tuna == 0 else q[r] for r, tuna in enumerate(y_t)]
        cost_1 = [0 if tuna == 1 else q[r] for r, tuna in enumerate(y_t)]
        reg0 = linear_model.LinearRegression()
        reg0.fit(x, cost_0)
        reg1 = linear_model.LinearRegression()
        reg1.fit(x, cost_1)
        primal_model = Reg_Oracle_Class.RegOracle(reg0, reg1)
        weighted_error = eval_weighted(q, x, y_t, primal_model)
        return [primal_model, weighted_error]
    if orc == 'svm':
        primal_model = svm.SVC()
    if orc == 'log':
        primal_model = linear_model.LogisticRegression()
    if orc == 'ada':
        primal_model = ensemble.AdaBoostClassifier()
    q_pos = [np.abs(z) for z in q]
    # flip weights to handle negative weights
    y_sign = [(1-y_t[j])*(1.0-np.sign(q[j]))/2.0 + y_t[j]*(1.0+np.sign(q[j]))/2.0 for j in range(len(y_t))]
    # weight normalization
    q_pos = np.multiply(1.0/np.mean(q_pos),q_pos)
    primal_model.fit(x, pd.DataFrame(y_sign).values.ravel(), sample_weight=q_pos)
    dummy_model = dummy.DummyClassifier()
    dummy_model.fit(x, [np.round(np.mean(y_t))] * len(y_t))
    weighted_error = eval_weighted(q, x, y_t, primal_model)
    return [primal_model, weighted_error]


def gen_sample(A):
    return [np.random.binomial(1, A[length], 1) for length in range(len(A))]

# update weights w to calculate the new p in the next iteration
def update_weights(w, A, group, X_prime, y, B, K, iteration):
    f_g = group
    # get group assignments for most recent group
    group_train = f_g.predict(X_prime)
    # compute proportion of 0's in population and in group
    n_0 = sum([1 for t in y if t == 0])
    n_g = sum([1 for i, t in enumerate(y) if t == 0 and group_train[i] == 1])
    C_s = B * np.sign(f[2] - FP)
    #print('C_t: {}'.format(C_s))
    # handle case when n_g = 0
    if n_g == 0:
        print('degenerate subgroup found')
        return np.multiply(w, (iteration - 1) / float(iteration))
    else:
        w_g = 1.0 / n_g
    if iteration > 1:
        psi = (iteration - 1) / iteration
    else:
        psi = 1
    w = np.multiply(w, psi)
    weights = [0, (1.0 / iteration) * C_s *
               (w_g - 1.0 / n_0), (-1.0 / iteration) * C_s * (1.0 / n_0)]
    #print(weights)
    # compute data point weights
    # use A_samp, alternatively could average updates over draws A_samp ~ A
    # get probabilistic decisions on X by empirical mixed strategy up to t-1
    for i in range(n):
        x_i = X_prime.iloc[i, :].values.reshape(1, -1)
        if y[i] == 1:
            w[i] += 0
        if y[i] == 0 and f_g.predict(pd.DataFrame(x_i))[0] == 1:
            w[i] += weights[1]
        if y[i] == 0 and f_g.predict(pd.DataFrame(x_i))[0] == 0:
            w[i] += weights[2]
    return w

# given a sequence of classifiers we want to print out the unfairness in each marginal coordinate
def calc_unfairness(A, X_prime, y_g, FP_p):
    unfairness = []
    n = X_prime.shape[1]
    sens_means = np.mean(X_prime, 0)
    for q in range(n):
        group_members = [X_prime.iloc[i, q] > sens_means[q] for i in range(X_prime.shape[0])]
        # calculate FP rate on group members
        fp_g = [a for t, a in enumerate(A) if group_members[t] == 1 and y_g[t] == 0]
        if len(fp_g) > 0:
            fp_g = np.mean(fp_g)
        else:
            fp_g = 0
        # calculate the fp rate on non-group members
        group_members_neg = [1-g for g in group_members]
        # calculate FP rate on group members
        fp_g_neg = [a for t, a in enumerate(A) if group_members_neg[t] == 1 and y_g[t] == 0]
        if len(fp_g_neg) > 0:
            fp_g_neg = np.mean(fp_g_neg)
        else:
            fp_g_neg = 0
        unfairness.append(np.max([np.abs(np.mean(fp_g)-FP_p), np.abs(np.mean(fp_g_neg)-FP_p)]))
    return unfairness


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
w = [0.0] * n
FP = 0
A = [0.0]*n
group_membership = [0.0]*n
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
    FP = ((iteration-1.0)/iteration)*FP + FP_new*(1.0/iteration)
    # dual player best responds: audit A via F, to get a group f: best
    # response to strategy up to t-1
    f = get_group(A, p, X, X_prime, y, K, aud_orc, FP)
    group_membership = np.add(group_membership, f[0].predict(X_prime))
    group_membership = [g != 0 for g in group_membership]
    # cumulative group members up to time t
    group_members_t = np.sum(group_membership)
    cum_group_mems.append(group_members_t)
    # primal player best responds: cost-sensitive classification
    # print(np.mean(w))
    w_prime = np.add(w, [1.0 / n] * n)
    #w_prime = np.multiply(w_prime, 1.0/np.mean(w_prime))
    #print('w_prime: {}.'.format(w_prime))
    p_t, lagrange = fit_weighted(w_prime, X, y, oracle)
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
    if aud_orc == 'log':
        coef_t.append(f[0].coef_)
    if aud_orc == 'reg_oracle':
        coef_t.append(f[0].b0.coef_ - f[0].b1.coef_)

    group_train = f[0].predict(X_prime)
    size_t.append(np.mean(group_train))
    if iteration == 1:
        print('most accurate classifier accuracy: {}, most acc-class unfairness: {}, most acc-class size {}'.format(err,fp_diff_t[0], size_t[0]))
    # get unfairness on marginal subgroups
    unfairness = calc_unfairness(A, X_prime, y, FP)
    # print
    if printflag:
        print('XX av error time {}, FP group diff, Group Size, Err Audit, FP Rate Diff Lag, Lgrgian err p_t, Cum_group: {} {} {} {} {} {} {}'.format(iteration, '{:f}'.format(err), '{:f}'.format(np.abs(f[1])), '{:f}'.format(np.mean(group_train)), '{:f}'.format(f[3]), '{:f}'.format(fp_rate_after_fit), '{:f}'.format(lagrange),'{:f}'.format(cum_group_mems[-1])))
        if aud_orc == 'log':
            tr = f[0].coef_[0]
            tr = [yt for yt in tr]
            t_str = str(tr).replace('\n', '')
            print('YYY coefficients of g_t: {}'.format(t_str),)
        if aud_orc == 'reg_oracle':
            group_coef = f[0].b0.coef_ - f[0].b1.coef_
            print('YYY coefficients of g_t: {}'.format(group_coef),)
        print('Unfairness in marginal subgroups: {}'.format(unfairness),)
    # update weights
    w = update_weights(w, A, f[0], X_prime, y, B, K, iteration)
    sys.stdout.flush()
    iteration += 1

# evaluate fair classifier found
D = gen_a(p[-1], X, y, A, iteration)
error_D = D[0]
model = fit_weighted([1.0 / n] * n, X, y, oracle)[0]
preds = model.predict(X)
error_opt = np.mean([np.abs(c - preds[i]) for i, c in enumerate(y)])
best_A = gen_a(model, X, y, [0.0]*n, iteration)
print('\n')
print('final classifier error on the data set: {}'.format(error_D))
print('best classifier error on the data set: {}'.format(error_opt))
print(
    'best classifier unfairness on the data set: {}'.format(
        fp_diff_t[0]))
print('FP base rate difference over time: {}'.format(fp_diff_t))
print('Classifier error over time: {}'.format(errors_t))
print('Group Size over time: {}'.format(size_t))

# if plotflag:
#     x = range(iteration)
#     plt.plot(x, errors_t, label='classifier error')
#     plt.plot(x, fp_diff_t, label='false positive difference in group')
#     plt.plot(x, size_t, label='size of group found')
#     plt.xlabel('iteration')
#     plt.legend()
#     plt.show()
