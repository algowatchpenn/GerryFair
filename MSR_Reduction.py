import sys
import clean_data_msr
import numpy as np
from sklearn import linear_model
import random
import Reg_Oracle_Class
random.seed(1)
# get command line arguments
num_sens, dataset, max_iters = sys.argv[1:]
num_sens = int(num_sens)
dataset = str(dataset)
max_iters = float(max_iters)

# Data Cleaning and Import
f_name = 'clean_{}'.format(dataset)
clean_the_dataset = getattr(clean_data_msr, f_name)
X, X_prime, y = clean_the_dataset(num_sens)
X, X_prime_cts, y = clean_the_dataset(num_sens)
n = X.shape[0]
# threshold sensitive features by average value
sens_means = np.mean(X_prime)
for col in X_prime.columns:
    X.loc[(X[col] > sens_means[col]), col] = 1
    X_prime.loc[(X_prime[col] > sens_means[col]), col] = 1
    X.loc[(X[col] <= sens_means[col]), col] = 0
    X_prime.loc[(X_prime[col] <= sens_means[col]), col] = 0


# Helper Functions
#
#
# calculate the next lambda via gradient descent
# for a group a compute P(preds = 1|A = a, Y = 0) = 1/n # y = 0, A = a,
# preds = 1 / p0a
def compute_fp(a, probs_0, preds, X_prime, X, y):
    p_0a = probs_0[a]
    y0_a = [y[i] == 0 and preds[i] == 1 and X_prime.iloc[i, a[0]] == a[1]
            for i in range(len(y))]
    p_y0_a = np.mean(y0_a)
    if p_0a == 0:
        return 'flag'
    return p_y0_a / p_0a


def next_lambda(lambda_0, iteration, probs_0, h, X_prime, X, y):
    A = lambda_0.keys()
    pred = h.predict(X)
    fp = {a: compute_fp(a, probs_0, pred, X_prime, X, y) for a in A}
    fp_av = np.mean({a: fp[a] for a in A if fp[a] != 'flag'}.values())
    for a in A:
        if fp[a] == 'flag':
            fp[a] = fp_av
    # update average vector with gradient. note gradient step is + eta*(grad)
    # since we are maximizing over lambda
    lambda_0 = {a: lambda_0[a] *
                (iteration -
                 1) /
                iteration +
                1.0 /
                iteration *
                (lambda_0[a] +
                 1.0 /
                 np.sqrt(iteration) *
                 (fp[a] -
                  fp_av)) for a in A}
    return lambda_0, [np.abs(fp_av - t) for t in fp.values()]


# lambda_0, lambda_1 : dictionaries whose keys are a in A, and values are lambda_ia
# probs_0, probs_1: dictionaries whose keys are a in A and values are P[A=a,y= y]
# return: cost arrays c_0 c_1
def update_costs(lambda_0, probs_0, X_prime, y):
    cost_0 = [float(y[i] != 0) for i in range(len(y))]
    cost_1_ind = [y[i] == 0 for i in range(len(y))]
    mu_0 = np.mean(lambda_0.values())
    weights_1 = [0.0] * len(y)
    for k in range(len(y)):
        for j in range(num_sens):
            a_kj = (j, X_prime.iloc[k, j])
            if a_kj in lambda_0.keys():
                y_k = y[k]
                if y_k == 0:
                    weights_1[k] += (lambda_0[a_kj] - mu_0) / probs_0[a_kj]
                else:
                    weights_1[k] = 0
    cost_1 = np.add(cost_1_ind, weights_1)
    return cost_0, cost_1


def fit_weighted(cost_0, cost_1, x):
    reg0 = linear_model.LinearRegression()
    reg0.fit(x, cost_0)
    reg1 = linear_model.LinearRegression()
    reg1.fit(x, cost_1)
    primal_model = Reg_Oracle_Class.RegOracle(reg0, reg1)
    return primal_model


def evaluate_classifier(h, X, y):
    preds = h.predict(X)
    acc = np.mean([preds[i] == y[i] for i in range(n)])
    return acc

# audit for FP unfairness using a logistic regression auditor


def fp_audit(A, df_sens, y):
    A_Y1 = [A[i] for i, c in enumerate(y) if c == 0]
    X_Y1 = [df_sens.iloc[i, :] for i, c in enumerate(y) if c == 0]
    clf_log_l1 = linear_model.LogisticRegressionCV(
        penalty='l1', solver='liblinear')
    clf_log_l1.fit(X_Y1, A_Y1)
    coef = clf_log_l1.coef_
    group_attributes = [df_sens.columns[i] for i in range(num_sens)]
    group_members = clf_log_l1.predict(df_sens)
    if np.sum(group_members) > 0:
        fp_group_rate = np.mean([A[i] for i, c in enumerate(
            group_members) if c == 1 and y[i] == 0])
        fp_base_rate = np.mean([A[i] for i, c in enumerate(y) if y[i] == 0])
        conjunction = np.sign(-1 * coef)
        print 'the fp rate in the group is {}'.format(fp_group_rate)
        print 'the fp base rate overall is: {}'.format(fp_base_rate)
        print 'the weight of the subgroup is: {}'.format(np.mean(group_members))
        print 'the relative error increase is: {}%'.format(100 * (fp_group_rate - fp_base_rate) / fp_base_rate)
        print 'group membership: {}'.format(group_members)
        print 'subgroup coefficients: {}'.format(coef)
        print('closest conjunction: {}*x < {}'.format(conjunction,
                                                      np.round(clf_log_l1.intercept_)))
        print 'the group attributes are: {}'.format(group_attributes)
    else:
        print('degenerate subgroup found: no unfairness')
    return group_members



# initialize parameters
iteration = 1
hypothesis = []
max_disp = []

# set up lambda, p dictionaries
immutable_keys = [(s, t) for s in range(num_sens) for t in [0, 1]]
# get unique keys
immutable_keys = set(immutable_keys)
# convert back to list (iterable)
immutable_keys = list(immutable_keys)


# set values of prob_0
prob_0 = {key: 1.0 / n * np.sum([X_prime.iloc[i, key[0]] == key[1]
                                 and y[i] == 0 for i in range(len(y))]) for key in immutable_keys}

# only focus on groups with non-trivial mass (speed up convergence)
alpha = .01
key_alpha = [key for key in prob_0.keys() if prob_0[key] > alpha]

lambda_0 = {el: 0 for el in key_alpha}
prob_0 = {key: prob_0[key] for key in key_alpha}

q = [1.0 / n] * n
cost_0 = [0 if s == 0 else q[r] for r, s in enumerate(y)]
cost_1 = [0 if s == 1 else q[r] for r, s in enumerate(y)]

h = fit_weighted(cost_0, cost_1, X)

# print out the best classifier error
h_dec = h.predict(X)
best_error = np.mean([h_dec[i] != y[i] for i in range(n)])
print('best classiifer error: {}'.format(best_error))


while iteration < max_iters:
    # primal player best responds via cost-sensitive learning oracle
    print(iteration)
    cost_0, cost_1 = update_costs(lambda_0, prob_0, X_prime, y)
    h = fit_weighted(cost_0, cost_1, X)
    # update lambda via gradient descent
    lambda_0, unfairness = next_lambda(
        lambda_0, iteration, prob_0, h, X_prime, X, y)
    hypothesis.append(h)
    max_disp.append(np.max(unfairness))
    acc = evaluate_classifier(h, X, y)
    print('fp_disparity in each group: {}'.format(unfairness),)
    print('max fp disparity in each group: {}'.format(np.max(unfairness)))
    print('error of h: {}'.format(1.0 - acc))
    A = h.predict(X)
    fp_audit(A, X_prime_cts, y)
    iteration += 1
# print the decisions of the final classifier
h = hypothesis[-1]
print('decisions on the dataset of the final fair classifier: {}'.format(h.predict(X)))

# audit the final classifier
