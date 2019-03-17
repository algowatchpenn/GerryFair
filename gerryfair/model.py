import numpy as np
import pandas as pd
from sklearn import linear_model
import random
import gerryfair.fairness_plots
import gerryfair.heatmap
from gerryfair.reg_oracle_class import RegOracle
import matplotlib.pyplot as plt

class Model:
    """Model object for fair learning and classification"""
    
    # Fictitious Play Algorithm
    # Input: dataset cleaned into X, X_prime, y, and arguments from commandline
    # Output: for each iteration the error and fairness violation - heatmap can also be produced
    def _fictitious_play(self,
                        X,
                        X_prime,
                        y):

        # defining variables and data structures for algorithm
        learner = Learner(X, y)
        auditor = Auditor(X_prime, y, self.fairness_def)

        n = X.shape[0]
        m = len([s for s in y if s == 0])

        # set default costs
        if self.fairness_def == 'FP':
            costs_0 = [0.0] * n
            costs_1 = [1.0 / n] * n
        elif self.fairness_def == 'FN':
            costs_1 = [0.0] * n
            costs_0 = [1.0 / n] * n

        p = [learner.best_response(costs_0, costs_1)]
        iteration = 1
        errors_t = []
        fairness_violations_t = []
        coef_t = []
        size_t = []
        groups = []
        cum_group_mems = []
        m = len([s for s in y if s == 0])
        metric_baseline = 0
        predictions = [0.0] * n
        group_membership = [0.0] * n
        X_0 = pd.DataFrame([X_prime.iloc[u, :] for u, s in enumerate(y) if s == 0])

        # scaling variables for heatmap
        vmin = None
        vmax = None

        while iteration < self.max_iters:
            print('iteration: {}'.format(int(iteration)))
            # get t-1 mixture decisions on X by randomizing on current set of p
            emp_p = learner.generate_predictions(p[-1], predictions, iteration)
            # get the error of the t-1 mixture classifier
            err = emp_p[0]
            # Average decisions
            predictions = emp_p[1]

            # update FP to get the false positive rate of the mixture classifier
            predictions_recent = p[-1].predict(X)

            # fairness metric baseline rate of t-1 mixture on new group g_t
            metric_baseline_recent = self.get_baseline(y, predictions) 
            metric_baseline = ((iteration - 1.0) / iteration) * metric_baseline + metric_baseline_recent * (1.0 / iteration)
            
            # dual player best responds to strategy up to t-1
            f = auditor.get_group(predictions, metric_baseline)

            group_size_0 = np.sum(f[0].predict(X_0)) * (1.0 / n)

            # compute list of people who have been included in an identified subgroup up to time t
            group_membership = np.add(group_membership, f[0].predict(X_prime))
            group_membership = [g != 0 for g in group_membership]
            group_members_t = np.sum(group_membership)
            cum_group_mems.append(group_members_t)

            # compute learner's best response to the CSC problem

            p_t = learner.best_response(costs_0, costs_1)
            A_t = p_t.predict(X)

            # append new group, new p, fairness_violations of group found, coefficients, group size
            groups.append(f[0])
            p.append(p_t)
            fairness_violations_t.append(np.abs(f[1]))
            errors_t.append(err)
            coef_t.append(f[0].b0.coef_ - f[0].b1.coef_)

            # outputs
            if iteration == 1:
                print(
                    'most accurate classifier accuracy: {}, most acc-class unfairness: {}, most acc-class size {}'.format(
                        err,
                        fairness_violations_t[0],
                        group_size_0))

            if self.printflag:
                print(
                    'ave error: {}, gamma-unfairness: {}, group_size: {}, frac included ppl: {}'.format('{:f}'.format(err),
                                                                                                        '{:f}'.format(
                                                                                                            np.abs(f[1])),
                                                                                                        '{:f}'.format(
                                                                                                            group_size_0),
                                                                                                        '{:f}'.format(
                                                                                                            cum_group_mems[
                                                                                                                -1] / float(
                                                                                                                n))))
            # save heatmap every heatmap_iter iterations
            if self.heatmapflag and (iteration % self.heatmap_iter) == 0:
                
                A_heat = A
                # initial heat map
                X_prime_heat = X_prime.iloc[:, 0:2]
                eta = 0.1

                minmax = heatmap.heat_map(X, X_prime_heat, y, A_heat, eta, self.heatmap_path + '/heatmap_iteration_{}'.format(iteration), vmin, vmax)
                if iteration == 1:
                    vmin = minmax[0]
                    vmax = minmax[1]

            # update costs: the primal player best responds
            costs_0, costs_1 = auditor.update_costs(costs_0, costs_1, f, self.C, iteration, self.gamma)
            iteration += 1    

        self.classifiers = p
        return errors_t, fairness_violations_t

    def get_baseline(self, y, y_hat):
        if self.fairness_def == 'FP':
            return np.mean([y_hat[i] for i, c in enumerate(y) if c == 0])
        elif self.fairness_def == 'FN':
            return np.mean([(1 - y_hat[i]) for i, c in enumerate(y) if c == 1])
        elif self.fairness_def == 'SP':
            return np.mean([(1-c)*(y_hat[i]) + c*(1-y_hat[i]) for i, c in enumerate(y)])


    def predict(self, X):
        num_classifiers = len(self.classifiers)

        y_hat = None
        for c in self.classifiers: 
            new_preds = np.multiply(1.0 / num_classifiers, c.predict(X))
            if y_hat is None:
                y_hat = new_preds
            else:
                y_hat = np.add(y_hat, new_preds)
        return [1 if y > .5 else 0 for y in y_hat]

    def pareto(self, X, X_prime, y, gamma_list, C=10, max_iters=10):
        # Store errors and fp over time for each gamma
        all_errors = []
        all_fp = []
        self.C = C
        self.max_iters = max_iters
        for g in gamma_list:
            self.gamma = g
            errors_gt, fairness_violations_gt = self._fictitious_play(X, X_prime, y)
            print(errors_gt, fairness_violations_gt)
            all_errors.append(np.mean(errors_gt))
            all_violations.append(np.mean(fairness_violations_gt))
        plt.plot(all_errors, all_violations)
        plt.xlabel('error')
        plt.ylabel('unfairness (fairness_violations*size)')
        plt.title('error vs. unfairness: C = {}, max_iters = {}'.format(C, max_iters))
        plt.show()
        return (all_errors, all_violations)

    def train(self, X, X_prime, y, alg="fict"):
        if alg == "fict":
            err, fairness_violations = self._fictitious_play(X, X_prime, y)
            return err, fairness_violations
        else:
            print("Specified algorithm is invalid")
            return

    ''' A method to switch the options before training'''
    def set_options(self, C=None,
                        printflag=None,
                        heatmapflag=None,
                        heatmap_iter=None,
                        heatmap_path=None,
                        max_iters=None,
                        gamma=None,
                        fairness_def=None):
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
                        printflag=False,
                        heatmapflag=False,
                        heatmap_iter=10,
                        heatmap_path='.',
                        max_iters=10,
                        gamma=0.01,
                        fairness_def='FP'):
        self.C = C
        self.printflag = printflag
        self.heatmapflag = heatmapflag
        self.heatmap_iter = heatmap_iter
        self.heatmap_path = heatmap_path
        self.max_iters = max_iters
        self.gamma = gamma
        self.fairness_def = fairness_def

class Learner:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def best_response(self, costs_0, costs_1):
        """Solve the CSC problem for the learner."""

        reg0 = linear_model.LinearRegression()
        reg0.fit(self.X, costs_0)
        reg1 = linear_model.LinearRegression()
        reg1.fit(self.X, costs_1)
        func = RegOracle(reg0, reg1)
        return func


    # Inputs:
    # q: the most recent classifier found
    # A: the previous set of decisions (probabilities) up to time iter - 1
    # iteration: the number of iteration
    # Outputs:
    # error: the error of the average classifier found thus far (incorporating q)
    def generate_predictions(self, q, A, iteration):
        """Return the classifications of the average classifier at time iter."""

        new_preds = np.multiply(1.0 / iteration, q.predict(self.X))
        ds = np.multiply((iteration - 1.0) / iteration, A)
        ds = np.add(ds, new_preds)
        error = np.mean([np.abs(ds[k] - self.y[k]) for k in range(len(self.y))])
        return [error, ds]

class Auditor:
    """This is the Auditor class. It is used in the training algorithm to repeatedly find subgroups that break the
    fairness disparity constraint. You can also use it independently as a stand alone auditor."""
    def __init__(self, X_prime, y, fairness_def):
        self.X_prime = X_prime
        self.y = y
        self.fairness_def = fairness_def
        self.X_prime_0 = pd.DataFrame([self.X_prime.iloc[u, :] for u, s in enumerate(self.y) if s == 0])
        self.X_prime_1 = pd.DataFrame([self.X_prime.iloc[u, :] for u, s in enumerate(self.y) if s == 1])


    def get_baseline(self, y, y_hat):
        if self.fairness_def == 'FP':
            return np.mean([y_hat[i] for i, c in enumerate(y) if c == 0])
        elif self.fairness_def == 'FN':
            return np.mean([(1 - y_hat[i]) for i, c in enumerate(y) if c == 1])
        elif self.fairness_def == 'SP':
            return np.mean([(1-c)*(y_hat[i]) + c*(1-y_hat[i]) for i, c in enumerate(y)])

    def update_costs(self, c_0, c_1, f, C, iteration, gamma):
        """Recursively update the costs from incorrectly predicting 1 for the learner."""
        # store whether FP disparity was + or -
        pos_neg = f[4]
        n = len(self.y)
        if self.fairness_def == 'FP':
            g_members = f[0].predict(self.X_prime_0)
            m = self.X_prime_0.shape[0]
            g_weight = np.sum(g_members)*(1.0/float(m))
            for i in range(n):
                X_prime_0_index = 0
                if self.y[i] == 0:
                    new_group_cost = (1.0/n)*pos_neg*C*(1.0/iteration) * (g_weight - g_members[X_prime_0_index])
                    if np.abs(f[1]) < gamma:
                        new_group_cost = 0
                    c_1[i] = (c_1[i] - 1.0/n) * ((iteration-1.0)/iteration) + new_group_cost + 1.0/n
                    X_prime_0_index += 1
                else:
                    c_1[i] = -1.0/n
            print(c_1)

        elif self.fairness_def == 'FN':
            g_members = f[0].predict(self.X_prime_1)
            m = self.X_prime_1.shape[0]
            g_weight = np.sum(g_members)*(1.0/float(m))
            for i in range(n):
                in_group_count = 0
                if self.y[i] == 1:
                    new_group_cost = (1.0/n)*pos_neg*C*(1.0/iteration) * (g_weight - g_members[in_group_count])
                    if np.abs(f[1]) < gamma:
                        new_group_cost = 0
                    c_0[i] = (c_0[i] - 1.0/n) * ((iteration-1.0)/iteration) + new_group_cost + 1.0/n
                    in_group_count += 1
                else:
                    c_0[i] = -1.0/n

        elif self.fairness_def == 'SP':
            g_members = f[0].predict(self.X_prime)
            m = self.X_prime.shape[0]
            g_weight = np.sum(g_members)*(1.0/float(m))
            for i in range(n):
                new_group_cost = (1.0/n)*pos_neg*C*(1.0/iteration) * (g_weight - g_members[i])
                if np.abs(f[1]) < gamma:
                    new_group_cost = 0
                c_1[i] = (c_1[i] - 1.0/n) * ((iteration-1.0)/iteration) + new_group_cost + 1.0/n
        
        return c_0, c_1

    def get_subset(self, predictions):
        if self.fairness_def == 'FP':
            return self.X_prime_0, [a for u, a in enumerate(predictions) if self.y[u] == 0]
        elif self.fairness_def == 'FN':
            return self.X_prime_1, [a for u, a in enumerate(predictions) if self.y[u] == 1]
        elif self.fairness_def == 'SP':
            return self.X_prime, predictions

    def get_group(self, predictions, metric_baseline):
        """Given decisions on sensitive attributes, labels, and FP rate audit wrt
            to gamma unfairness. Return the group found, the gamma unfairness, fp disparity, and sign(fp disparity).
        """
        X_subset, predictions_subset = self.get_subset(predictions)

        m = len(predictions_subset)
        n = float(len(self.y))
        cost_0 = [0.0] * m
        cost_1 = -1.0 / n * (metric_baseline - predictions_subset)
        reg0 = linear_model.LinearRegression()
        reg0.fit(X_subset, cost_0)
        reg1 = linear_model.LinearRegression()
        reg1.fit(X_subset, cost_1)
        func = RegOracle(reg0, reg1)
        group_members_0 = func.predict(X_subset)
        err_group = np.mean([np.abs(group_members_0[i] - predictions_subset[i])
                             for i in range(len(predictions_subset))])
        # get the false positive rate in group
        if sum(group_members_0) == 0:
            fp_group_rate = 0
        else:
            fp_group_rate = np.mean([r for t, r in enumerate(predictions_subset) if group_members_0[t] == 1])
        g_size_0 = np.sum(group_members_0) * 1.0 / n
        fp_disp = np.abs(fp_group_rate - metric_baseline)
        fp_disp_w = fp_disp * g_size_0

        # negation
        cost_0_neg = [0.0] * m
        cost_1_neg = -1.0 / n * (predictions_subset - metric_baseline)
        reg0_neg = linear_model.LinearRegression()
        reg0_neg.fit(X_subset, cost_0_neg)
        reg1_neg = linear_model.LinearRegression()
        reg1_neg.fit(X_subset, cost_1_neg)
        func_neg = RegOracle(reg0_neg, reg1_neg)
        group_members_0_neg = func_neg.predict(X_subset)
        err_group_neg = np.mean(
            [np.abs(group_members_0_neg[i] - predictions_subset[i]) for i in range(len(predictions_subset))])
        if sum(group_members_0_neg) == 0:
            fp_group_rate_neg = 0
        else:
            # update 
            fp_group_rate_neg = np.mean([r for t, r in enumerate(predictions_subset) if group_members_0[t] == 0])
        g_size_0_neg = np.sum(group_members_0_neg) * 1.0 / n
        fp_disp_neg = np.abs(fp_group_rate_neg - metric_baseline)
        fp_disp_w_neg = fp_disp_neg*g_size_0_neg

        # return group
        if fp_disp_w_neg > fp_disp_w:
            return [func_neg, fp_disp_w_neg, fp_disp_neg, err_group_neg, -1]
        else:
            return [func, fp_disp_w, fp_disp, err_group, 1]

    def audit(self, predictions, X_prime, y):
        """Takes in predictions on dataset (X',y) and returns:
            a vector which represents the group that violates the fairness metric, along with the u.
        """
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values

        metric_baseline = self.get_baseline(y, predictions)
        aud_group, gamma_unfair, fp_in_group, err_group, pos_neg = self.get_group(predictions, metric_baseline)

        return aud_group.predict(X_prime), gamma_unfair
