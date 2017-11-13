# to do: discard some columns to enlarge dataset
import sys
import numpy as np
import pandas as pd
from sklearn import linear_model
import random
random.seed(1)
num_sens = sys.argv[1:][0]
print('number of sensitive attributes: {}, random seed = 1'.format(num_sens))
num_sens = int(num_sens)
df = pd.read_csv('communities.csv')
df = df.fillna(0)
n_test = 300
df_test = df.iloc[1:n_test, :]
df = df.iloc[(1 + n_test):, :]
sens_features = [3,4,5,6,22,23,24,25,26,27,61,62,92,105,106,107,108,109]
df_sens = df.iloc[:, sens_features[0:num_sens]]
df_sens_test = df_test.iloc[:, sens_features[0:num_sens]]
y = df['ViolentCrimesPerPop']
y_test = df_test['ViolentCrimesPerPop']
q_y = np.percentile(y, 70)
# convert y's to binary predictions on whether the neighborhood is
# especially violent
y = [np.round((1 + np.sign(s - q_y)) / 2) for s in y]
y_test = [np.round((1 + np.sign(s - q_y)) / 2) for s in y_test]

# train a logistic regression model to predict Y
clf = linear_model.LogisticRegressionCV(penalty='l2')
X = df.iloc[:, 0:122]
clf.fit(X, y)
A = clf.predict(X)

# get test predictions
X_test = df_test.iloc[:, 0:122]
A_test = clf.predict(X_test)

# Statistical Parity
# train linear classifier to predict A via df_sense (use lasso)


def sp_audit(A, df_sens, A_test, df_sens_test):
    clf_log_l1 = linear_model.LogisticRegressionCV(
        penalty='l1', solver='liblinear')
    clf_log_l1.fit(df_sens, A)
    coef = clf_log_l1.coef_
    coef_nonzero = [1 * (np.abs(c) > 0) for c in coef]
    coef_nonzero = coef_nonzero[0]
    group_attributes = [df_sens.columns[i]
                        for i, c in enumerate(coef_nonzero) if c == 1]
    # get the group members in the test set
    group_members = clf_log_l1.predict(df_sens_test)
    # compute base rate and group rate in test set
    base_rate_group = np.mean([A_test[i]
                               for i, c in enumerate(group_members) if c == 1])
    print 'stat parity base rate in the group is {}'.format(base_rate_group)
    print 'the stat parity base rate overall is: {}'.format(np.mean(A))
    print 'the weight of the subgroup is: {}'.format(np.mean(group_members))
    # attributes with positive coefficients (i.e. that increase the
    # probability of mis-classification)
    print 'the negative group attributes are: {}'.format(group_attributes)
    print 'group membership: {}'.format(group_members)
    print 'subgroup coefficients: {}'.format(coef)
    return group_members


def fp_audit(A, df_sens, df_sens_test, A_test, y, y_test):
    A_Y1 = [A[i] for i, c in enumerate(y) if c == 0]
    X_Y1 = [df_sens.iloc[i, :] for i, c in enumerate(y) if c == 0]
    clf_log_l1 = linear_model.LogisticRegressionCV(
        penalty='l1', solver='liblinear')
    clf_log_l1.fit(X_Y1, A_Y1)
    coef = clf_log_l1.coef_
    group_attributes = [df_sens.columns[i] for i in range(num_sens)]
    group_members = clf_log_l1.predict(df_sens_test)
    fp_group_rate = np.mean([A_test[i] for i, c in enumerate(
        group_members) if c == 1 and y_test[i] == 0])
    fp_base_rate = np.mean([A_test[i]
                            for i, c in enumerate(y_test) if y_test[i] == 0])
    conjunction = np.sign(-1*coef)
    print 'the fp rate in the group is {}'.format(fp_group_rate)
    print 'the fp base rate overall is: {}'.format(fp_base_rate)
    print 'the weight of the subgroup is: {}'.format(np.mean(group_members))
    print 'the relative error increase is: {}%'.format(100 * (fp_group_rate - fp_base_rate) / fp_base_rate)
    print 'group membership: {}'.format(group_members)
    print 'subgroup coefficients: {}'.format(coef)
    print('closest conjunction: {}*x < {}'.format(conjunction, np.round(clf_log_l1.intercept_)))
    print 'the group attributes are: {}'.format(group_attributes)
    return group_members


sp_audit(A, df_sens, A_test, df_sens_test)
fp_audit(A, df_sens, df_sens_test, A_test, y, y_test)
