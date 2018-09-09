# Modified based on expgrad algorithm:
# https://github.com/Microsoft/fairlearn

from __future__ import print_function

import functools
import numpy as np
import pandas as pd
import fairlearn.moments as moments
import fairlearn.classred as red
import clean_data_msr as parser2
import clean_data as parser1
import Audit as audit
import pickle
from sklearn import linear_model

print = functools.partial(print, flush=True)


class LeastSquaresLearner:    
    def __init__(self):
        self.weights = None
        
    def fit(self, X, Y, W):
        sqrtW = np.sqrt(W)
        matX = np.array(X) * sqrtW[:, np.newaxis]
        vecY = Y * sqrtW
        self.lsqinfo = np.linalg.lstsq(matX, vecY, rcond=-1)
        self.weights = pd.Series(self.lsqinfo[0], index=list(X))

    def predict(self, X):
        pred = X.dot(self.weights)
        return 1*(pred>0.5)


class RegressionLearner:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y, W):
        cost_vec0 = Y * W  # cost vector for predicting zero
        cost_vec1 = (1 - Y) * W
        self.reg0 = linear_model.LinearRegression()
        self.reg0.fit(X, cost_vec0)
        self.reg1 = linear_model.LinearRegression()
        self.reg1.fit(X, cost_vec1)

    def predict(self, X):
        pred0 = self.reg0.predict(X)
        pred1 = self.reg1.predict(X)
        return 1*(pred1 < pred0)

class DP_named(moments.DP):
    """Demo parity with a named attribute"""
    short_name = "DP_named"

    def __init__(self, attr):
        super().__init__()
        self.attr = attr

    def init(self, dataX, dataA, dataY):
        super().init(dataX, dataA[self.attr], dataY)

class DP_2attrs():
    """Demo parity with marginals on two attributes; for testing"""
    short_name = "DP_2attrs"
    
    def __init__(self, attr1, attr2):
        self.attr1 = attr1
        self.attr2 = attr2
        self.DP1 = DP_named(attr1)
        self.DP2 = DP_named(attr2)
        
    def init(self, dataX, dataA, dataY):
        self.DP1.init(dataX, dataA, dataY)
        self.DP2.init(dataX, dataA, dataY)
        dummy_gamma = self.gamma(lambda X: 0)
        self.index = dummy_gamma.index

    def gamma(self, predictor):
        gamma1 = self.DP1.gamma(predictor)
        gamma2 = self.DP2.gamma(predictor)
        return pd.concat([gamma1, gamma2], keys=[self.attr1, self.attr2])

    def lambda_signed(self, lambda_vec):
        lambda1 = self.DP1.lambda_signed(lambda_vec[self.attr1])
        lambda2 = self.DP2.lambda_signed(lambda_vec[self.attr2])
        return pd.concat([lambda1, lambda2], keys=[self.attr1, self.attr2])

    def signed_weights(self, lambda_vec):
        sw1 = self.DP1.signed_weights(lambda_vec[self.attr1])
        sw2 = self.DP2.signed_weights(lambda_vec[self.attr2])
        return sw1+sw2

class marginal_DP():
    """Demo parity with marginals on attributes"""
    short_name = "marginal_DP"

    def __init__(self, attr_list):
        num_attr = len(attr_list)
        self.attr_list = attr_list
        self.DP = {}
        for i in range(num_attr):
            self.DP[i] = DP_named(attr_list[i])

    def init(self, dataX, dataA, dataY):
        num_attr = len(self.attr_list)
        for i in range(num_attr):
            self.DP[i].init(dataX, dataA, dataY)
        dummy_gamma = self.gamma(lambda X: 0)
        self.index = dummy_gamma.index

    def gamma(self, predictor):
        num_attr = len(self.attr_list)
        gamma = {}
        for i in range(num_attr):
            gamma[i] = self.DP[i].gamma(predictor)
        return pd.concat(list(gamma.values()), keys=self.attr_list)

    def lambda_signed(self, lambda_vec):
        num_attr = len(self.attr_list)
        lambda_dict = {}
        for i in range(num_attr):
            lambda_dict[i] = self.DP[i].lambda_signed(lambda_vec[self.attr_list[i]])
        return pd.concat(list(lambda_dict.values()), keys=self.attr_list)

    def signed_weights(self, lambda_vec):
        num_attr = len(self.attr_list)
        sw_dict = {}
        for i in range(num_attr):
            sw_dict[i] = self.DP[i].signed_weights(lambda_vec[self.attr_list[i]])
        return sum(list(sw_dict.values()))

class EO_named(moments.EO):
    """Demo parity with a named attribute"""
    short_name = "EO_named"

    def __init__(self, attr):
        super().__init__()
        self.attr = attr

    def init(self, dataX, dataA, dataY):
        super().init(dataX, dataA[self.attr], dataY)

class EO_2attrs():
    """Demo parity with marginals on two attributes; for testing"""
    short_name = "EO_2attrs"
    
    def __init__(self, attr1, attr2):
        self.attr1 = attr1
        self.attr2 = attr2
        self.EO1 = EO_named(attr1)
        self.EO2 = EO_named(attr2)
        
    def init(self, dataX, dataA, dataY):
        self.EO1.init(dataX, dataA, dataY)
        self.EO2.init(dataX, dataA, dataY)
        dummy_gamma = self.gamma(lambda X: 0)
        self.index = dummy_gamma.index

    def gamma(self, predictor):
        gamma1 = self.EO1.gamma(predictor)
        gamma2 = self.EO2.gamma(predictor)
        return pd.concat([gamma1, gamma2], keys=[self.attr1, self.attr2])

    def lambda_signed(self, lambda_vec):
        lambda1 = self.EO1.lambda_signed(lambda_vec[self.attr1])
        lambda2 = self.EO2.lambda_signed(lambda_vec[self.attr2])
        return pd.concat([lambda1, lambda2], keys=[self.attr1, self.attr2])

    def signed_weights(self, lambda_vec):
        sw1 = self.EO1.signed_weights(lambda_vec[self.attr1])
        sw2 = self.EO2.signed_weights(lambda_vec[self.attr2])
        return sw1+sw2

class marginal_EO():
    """Demo parity with marginals on attributes"""
    short_name = "marginal_EO"

    def __init__(self, attr_list):
        num_attr = len(attr_list)
        self.attr_list = attr_list
        self.EO = {}
        for i in range(num_attr):
            self.EO[i] = EO_named(attr_list[i])

    def init(self, dataX, dataA, dataY):
        num_attr = len(self.attr_list)
        for i in range(num_attr):
            self.EO[i].init(dataX, dataA, dataY)
        dummy_gamma = self.gamma(lambda X: 0)
        self.index = dummy_gamma.index

    def gamma(self, predictor):
        num_attr = len(self.attr_list)
        gamma = {}
        for i in range(num_attr):
            gamma[i] = self.EO[i].gamma(predictor)
        return pd.concat(list(gamma.values()), keys=self.attr_list)

    def lambda_signed(self, lambda_vec):
        num_attr = len(self.attr_list)
        lambda_dict = {}
        for i in range(num_attr):
            lambda_dict[i] = self.EO[i].lambda_signed(lambda_vec[self.attr_list[i]])
        return pd.concat(list(lambda_dict.values()), keys=self.attr_list)

    def signed_weights(self, lambda_vec):
        num_attr = len(self.attr_list)
        sw_dict = {}
        for i in range(num_attr):
            sw_dict[i] = self.EO[i].signed_weights(lambda_vec[self.attr_list[i]])
        return sum(list(sw_dict.values()))

def weighted_predictions(res_tuple, x):
    """
    Given res_tuple from expgrad, compute the weighted predictions
    over the dataset x
    """
    hs = res_tuple.classifiers
    weights = res_tuple.weights  # weights over classifiers
    preds = hs.apply(lambda h: h.predict(x))  # predictions
    # return weighted predictions
    return weights.dot(preds)


def binarize_sens_attr(a):
    """
    given a set of sensitive attributes; binarize them
    """
    for col in a.columns:
        if len(a[col].unique()) > 2:  # hack: identify numeric features
            sens_mean = np.mean(a[col])
            a[col] = 1 * (a[col] > sens_mean)



def print_marginal_avg_pred(x, a, res_tuple):
    for j in range(len(a.values[0])):
        aj = a.iloc[:, j]
        xj = x[aj == 1]
        w_predj = weighted_predictions(res_tuple, xj)
        print('avg prediction for ', j, sum(w_predj)/len(w_predj))


def run_eps_list_FP(eps_list, dataset):
    if dataset == 'communities':
        x, a, y = parser1.clean_communities()
    elif dataset == 'communities2':
        x, a, y = parser2.clean_communities(18)
    elif dataset == 'student':
        x, a, y = parser1.clean_student()
    elif dataset == 'lawschool':
        x, a, y = parser1.clean_lawschool()
    elif dataset == 'adult':
        x, a, y = parser1.clean_adult()
    else:
        raise Exception('Dataset not in range!')

    learner = RegressionLearner()
    a_prime = a.copy(deep=True)
    binarize_sens_attr(a_prime)
    a_prime[y == 1] = 0  # hack: setting protected attrs to be 0 for
                         # negative examples
    sens_attr = list(a_prime.columns)

    gamma_values = {}
    err_values = {}
    eps_values = {}
    for eps in eps_list:
        res_tuple = red.expgrad(x, a_prime, y, learner,
                                cons=marginal_EO(sens_attr), eps=eps)
        weighted_pred = weighted_predictions(res_tuple, x)
        err_values[eps] = sum(np.abs(y - weighted_pred)) / len(y)  # err 
        gamma_values[eps] = audit.audit(weighted_pred, x, a, y)    # gamma
        eps_values[eps] = compute_FP(a_prime, y, weighted_pred)
        print(eps_values[eps])
    d = {'err' : list(err_values.values()), 'gamma' :
         list(gamma_values.values()), 'input eps' : eps_list,
         'empirical eps' : list(eps_values.values())}
    return pd.DataFrame(data=d)



def compute_EO(a, y, weighted_pred):
    """
    Debug fn: compute equalized odds given weighted_pred
    assume a is already binarized
    """
    sens_attr = list(a.columns)
    disp = {}
    for c in sens_attr:
        for y_val in [0, 1]:
            for a_val in [0, 1]:
                a_c = a[c]
                # calculate Pr[ y-hat = 1 | y = 1 ]
                p_all = np.average(weighted_pred[y == y_val])

                if len(weighted_pred[(y == y_val) & (a_c == a_val)]) > 0:
                    # calculate Pr[ y-hat = 1 | y = 1, a=1]
                    p_sub = np.average(weighted_pred[(y == y_val) & (a_c == a_val)])
                    disp[(c, y_val, a_val)] = np.abs(p_all - p_sub)
    eps = max(disp.values())
    # (c_max, a_max, _) = max(disp, key=disp.get)
    # group_size = len(y[a[c_max] == a_max]) / len(y)
    return eps



def compute_FP(a, y, weighted_pred):
    """
    Debug fn: compute FP disp given weighted_pred
    assume a is already binarized
    """
    sens_attr = list(a.columns)
    disp = {}
    for c in sens_attr:
        for a_val in [0, 1]:
            a_c = a[c]
            # calculate Pr[ y-hat = 1 | y = 1 ]
            p_all = np.average(weighted_pred[y == 0])

            if len(weighted_pred[(y == 0) & (a_c == a_val)]) > 0:
                # calculate Pr[ y-hat = 1 | y = 1, a=1]
                p_sub = np.average(weighted_pred[(y == 0) & (a_c == a_val)])
                disp[(c, a_val)] = np.abs(p_all - p_sub)
    return max(disp.values())



def run_eps_single(eps, dataset):
    """Running the expgrad algorithm once, and then audit for subgroup
fairness

    """
    if dataset == 'communities':
        x, a, y = parser1.clean_communities()
    elif dataset == 'communities2':
        x, a, y = parser2.clean_communities(18)
    elif dataset == 'student':
        x, a, y = parser1.clean_student()
    elif dataset == 'lawschool':
        x, a, y = parser1.clean_lawschool()
    elif dataset == 'adult':
        x, a, y = parser1.clean_adult()
    else:
        raise Exception('Dataset not in range!')

    learner0 = RegressionLearner()
    # directly running oracle for comparison
    learner0.fit(x, y, np.ones(len(y)))
    pred0 = learner0.predict(x)
    err0 = sum(np.abs(y - pred0)) / len(y)
    print('Base error rate: ', err0)

    learner = RegressionLearner()
    a_prime = a.copy(deep=True)
    binarize_sens_attr(a_prime)
    a_prime[y == 1] = 0
    sens_attr = list(a.columns)
    res_tuple = red.expgrad(x, a_prime, y, learner,
                            cons=marginal_EO(sens_attr), eps=eps,
                            debug=True)
    weighted_pred = weighted_predictions(res_tuple, x)
    gamma = audit.audit(weighted_pred, x, a, y)

    print('gamma ', gamma)
    err = sum(np.abs(y - weighted_pred)) / len(y)  # err
    print('err ', err)
    print('EO: ', compute_EO(a_prime, y, weighted_pred))
    binarize_sens_attr(a)
    print('FP: ', compute_FP(a, y, weighted_pred))





base_eps_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
            0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09,
            0.1, 0.2, 0.4, 0.8, 1]

eps_list = list(sorted(set(base_eps_list + list(np.linspace(0.01, 0.2, 51)))))


data_list = ['student', 'communities', 'adult', 'lawschool']

for dataset in data_list:
    print('Current dataset: ' + dataset)
    data = run_eps_list_FP(eps_list, dataset)
    pickle.dump(data, open(dataset+'_fp_exp.p', 'wb'))
