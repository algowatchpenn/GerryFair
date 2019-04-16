import numpy as np
import pandas as pd
from sklearn import linear_model
import gerryfair.fairness_plots
import gerryfair.heatmap
from gerryfair.learner import Learner
from gerryfair.auditor import Auditor
from gerryfair.classifier_history import ClassifierHistory
from gerryfair.reg_oracle_class import RegOracle
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Model:
    """Model object for fair learning and classification"""

    def fictitious_play(self,

                        X,
                        X_prime,
                        y,
                        early_termination=True):
        """
        Fictitious Play Algorithm
        Input: dataset cleaned into X, X_prime, y
        Output: for each iteration the error and fairness violation - heatmap can also be produced. classifiers stored in class state.
        """

        # defining variables and data structures for algorithm
        learner = Learner(X, y, self.predictor)
        auditor = Auditor(X_prime, y, self.fairness_def)
        history = ClassifierHistory()

        # initialize variables
        n = X.shape[0]
        costs_0, costs_1, X_0 = auditor.initialize_costs(n)
        metric_baseline = 0
        predictions = [0.0] * n

        # scaling variables for heatmap
        vmin = None
        vmax = None

        # print output variables
        errors = []
        fairness_violations = []

        iteration = 1
        while iteration < self.max_iters:
            # learner's best response: solve the CSC problem, get mixture decisions on X to update prediction probabilities
            history.append_classifier(learner.best_response(costs_0, costs_1)) 
            (error, predictions) = learner.generate_predictions(history.get_most_recent(), predictions, iteration)
            
            # auditor's best response: find group, update costs
            metric_baseline = auditor.get_baseline(y, predictions) 
            group = auditor.get_group(predictions, metric_baseline)
            costs_0, costs_1 = auditor.update_costs(costs_0, costs_1, group, self.C, iteration, self.gamma)

            # outputs
            errors.append(error)
            fairness_violations.append(group.weighted_disparity)
            self.print_outputs(iteration, error, group)
            vmin, vmax = self.save_heatmap(iteration, X, X_prime, y, history.get_most_recent().predict(X), vmin, vmax)
            iteration += 1

            # early termination:
            if early_termination and (len(errors) >= 5) and ((errors[-1] == errors[-2]) or fairness_violations[-1] == fairness_violations[-2]) and fairness_violations[-1] < self.gamma:
                iteration = self.max_iters

        self.classifiers = history.classifiers
        return errors, fairness_violations

    def print_outputs(self, iteration, error, group):
        print('iteration: {}'.format(int(iteration)))
        if iteration == 1:
            print(
                'most accurate classifier error: {}, most accurate class unfairness: {}, violated group size: {}'.format(
                    error,
                    group.weighted_disparity,
                    group.group_size))

        elif self.printflag:
            print(
                'error: {}, fairness violation: {}, violated group size: {}'.format(
                    error,
                    group.weighted_disparity,
                    group.group_size))

    
    def save_heatmap(self, iteration, X, X_prime, y, predictions, vmin, vmax):
        '''Helper method: save heatmap frame'''

        # save heatmap every heatmap_iter iterations
        if self.heatmapflag and (iteration % self.heatmap_iter) == 0:
            # initial heat map
            X_prime_heat = X_prime.iloc[:, 0:2]
            eta = 0.1
            minmax = heatmap.heat_map(X, X_prime_heat, y, predictions_t, eta, self.heatmap_path + '/heatmap_iteration_{}'.format(iteration), vmin, vmax)
            if iteration == 1:
                vmin = minmax[0]
                vmax = minmax[1]
        return vmin, vmax

    
    def predict(self, X):
        ''' Generates predictions. We do not yet advise using this in sensitive real-world settings. '''

        num_classifiers = len(self.classifiers)
        y_hat = None
        for c in self.classifiers: 
            new_preds = np.multiply(1.0 / num_classifiers, c.predict(X))
            if y_hat is None:
                y_hat = new_preds
            else:
                y_hat = np.add(y_hat, new_preds)
        return [1 if y > .5 else 0 for y in y_hat]


    def pareto(self, X, X_prime, y, gamma_list):
        '''Assumes Model has FP specified for metric. 
        Trains for each value of gamma, returns error, FP (via training), and FN (via auditing) values.'''

        C=self.C
        max_iters=self.max_iters
        # Store errors and fp over time for each gamma

        # change var names, but no real dependence on FP logic
        all_errors = []
        all_fp_violations = []
        all_fn_violations = []
        self.C = C
        self.max_iters = max_iters

        auditor = Auditor(X_prime, y, 'FN')
        for g in gamma_list:
            self.gamma = g
            errors, fairness_violations = self.train(X, X_prime, y)
            predictions = self.predict(X)
            _, fn_violation = auditor.audit(predictions)
            all_errors.append(errors_gt[-1])
            all_fp_violations.append(fairness_violations[-1])
            all_fn_violations.append(fn_violation)

        return (all_errors, all_fp_violations, all_fn_violations)

    
    def train(self, X, X_prime, y, alg="fict"):
        ''' Trains a subgroup-fair model using provided data and specified parameters. '''

        if alg == "fict":
            err, fairness_violations = self.fictitious_play(X, X_prime, y)
            return err, fairness_violations
        else:
            raise Exception("Specified algorithm is invalid")

    
    def set_options(self, C=None,
                        printflag=None,
                        heatmapflag=None,
                        heatmap_iter=None,
                        heatmap_path=None,
                        max_iters=None,
                        gamma=None,
                        fairness_def=None):
        ''' A method to switch the options before training. '''

        if C:
            self.C = C
        if printflag:
            self.printflag = printflag
        if heatmapflag:
            self.heatmapflag = heatmapflag
        if heatmap_iter:
            self.heatmap_iter = heatmap_iter
        if heatmap_path:
            self.heatmap_path = heatmap_path
        if max_iters:
            self.max_iters = max_iters
        if gamma:
            self.gamma = gamma
        if fairness_def:
            self.fairness_def = fairness_def


    def __init__(self, C=10,
                        metric='FP',
                        printflag=False,
                        heatmapflag=False,
                        heatmap_iter=10,
                        heatmap_path='.',
                        max_iters=10,
                        gamma=0.01,
                        fairness_def='FP',
                        predictor=linear_model.LinearRegression()):
        self.C = C
        self.metric = metric
        self.printflag = printflag
        self.heatmapflag = heatmapflag
        self.heatmap_iter = heatmap_iter
        self.heatmap_path = heatmap_path
        self.max_iters = max_iters
        self.gamma = gamma
        self.fairness_def = fairness_def
        self.predictor = predictor
        if self.fairness_def not in ['FP', 'FN']:
            raise Exception('This metric is not yet supported for learning. Metric specified: {}.'.format(self.fairness_def))

<<<<<<< HEAD
class Learner:
    def __init__(self, X, y):
        self.X = X
        self.y = y


    def best_response(self, c_1t):
        # should be passed c_0, c_1

        """Solve the CSC problem for the learner."""
        n = len(self.y)
        c_1t_new = c_1t[:]
        c_0 = [0.0] * n
        c_1 = []
        for r in range(n):
            if self.y[r] == 1:
                c_1.append((-1.0/n))
            else:
                c_1.append(c_1t_new.pop(0))
        reg0 = linear_model.LinearRegression()
        reg0.fit(self.X, c_0)
        reg1 = linear_model.LinearRegression()
        reg1.fit(self.X, c_1)
        func = RegOracle(reg0, reg1)
        return func


    # Inputs:
    # A: the previous set of decisions (probabilities) up to time iter - 1
    # q: the most recent classifier found
    # x: the dataset
    # y: the labels
    # iter: the iteration
    # Outputs:
    # error: the error of the average classifier found thus far (incorporating q)
    def generate_predictions(self, q, prev_decisions, iteration):
        """Return the classifications of the average classifier at time iter."""

        new_preds = np.multiply(1.0 / iteration, q.predict(self.X))
        old_preds = np.multiply((iteration - 1.0) / iteration, prev_decisions)
        preds_mixture = np.add(old_preds, new_preds)
        error = np.mean([np.abs(preds_mixture[k] - self.y[k]) for k in range(len(self.y))])
        return [error, ds]

class Auditor:
    """docstring for Auditor"""
    def update_costs(self, c_1, f, X_prime, y, C, iteration, fp_disp, gamma):
        """Recursively update the costs from incorrectly predicting 1 for the learner."""
        # store whether FP disparity was + or - (UPDATE for FN/SP)
        pos_neg = f[4]
        X_0_prime = pd.DataFrame([X_prime.iloc[u, :] for u,s in enumerate(y) if s == 0])
        g_members = f[0].predict(X_0_prime)
        m = len(c_1)
        n = float(len(y))
        g_weight_0 = np.sum(g_members)*(1.0/float(m))
        for t in range(m):
            new_group_cost = (1.0/n)*pos_neg*C*(1.0/iteration) * (g_weight_0 - g_members[t])
            if np.abs(fp_disp) < gamma:
                if t == 0:
                    print('barrier')
                new_group_cost = 0
            c_1[t] = (c_1[t] - 1.0/n) * ((iteration-1.0)/iteration) + new_group_cost + 1.0/n
        return c_1

    def get_group(self, A, X_sens, y_g, FP):
        """Given decisions on sensitive attributes, labels, and FP rate audit wrt
            to gamma unfairness. Return the group found, the gamma unfairness, fp disparity, and sign(fp disparity).

            (UPDATE for FN/SP)
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
        func = RegOracle(reg0, reg1)
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
        func_neg = RegOracle(reg0_neg, reg1_neg)
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

    def audit(self, predictions, X_prime, y):
        """Takes in predictions on dataset (X',y) and prints gamma-unfairness,
        fp disparity, group size, group coefficients, and sensitive column names.

        (UPDATE for FN/SP)
        """
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values
        FP = np.mean([p for i,p in enumerate(predictions) if y[i] == 0])
        aud_group, gamma_unfair, fp_in_group, err_group, pos_neg = self.get_group(predictions, X_sens=X_prime, y_g=y, FP=FP)
=======
>>>>>>> staging

