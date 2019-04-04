import numpy as np
import pandas as pd
from sklearn import linear_model
import gerryfair.fairness_plots
import gerryfair.heatmap
from gerryfair.learner import Learner
from gerryfair.auditor import Auditor
from gerryfair.reg_oracle_class import RegOracle
import matplotlib

matplotlib.use('TkAgg')
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
        learner = Learner(X, y, self.predictor)
        auditor = Auditor(X_prime, y, self.fairness_def)

        n = X.shape[0]
        m = len([s for s in y if s == 0])

        # set default costs
        if self.fairness_def == 'FP':
            costs_0 = [0.0] * n
            costs_1 = [-1.0 / n * (2 * i - 1) for i in y]
            X_0 = pd.DataFrame([X_prime.iloc[u, :] for u, s in enumerate(y) if s == 0])
        elif self.fairness_def == 'FN':
            costs_0 = [0.0] * n
            costs_1 = [1.0 / n * (2 * i - 1) for i in y]
            X_0 = pd.DataFrame([X_prime.iloc[u, :] for u, s in enumerate(y) if s == 1])

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
            metric_baseline_recent = auditor.get_baseline(y, predictions) 
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

            # early termination:
            if (len(errors_t) >= 5) and ((errors_t[-1] == errors_t[-2]) or fairness_violations_t[-1] == fairness_violations_t[-2]) and fairness_violations_t[-1] < self.gamma:
                iteration = self.max_iters

        self.classifiers = p
        return errors_t, fairness_violations_t

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

    def pareto(self, X, X_prime, y, gamma_list):
        C=self.C
        max_iters=self.max_iters
        # Store errors and fp over time for each gamma
        all_errors = []
        all_fp_violations = []
        all_fn_violations = []
        self.C = C
        self.max_iters = max_iters

        auditor = Auditor(X_prime, y, 'FN')
        for g in gamma_list:
            self.gamma = g
            errors_gt, fairness_violations_gt = self.train(X, X_prime, y)
            predictions = self.predict(X)
            _, fn_violation = auditor.audit(predictions)
            all_errors.append(errors_gt[-1])
            all_fp_violations.append(fairness_violations_gt[-1])
            all_fn_violations.append(fn_violation)
        '''
        plt.plot(all_errors, all_violations)
        plt.xlabel('error')
        plt.ylabel('fairness violation')
        plt.title('error vs. unfairness: C = {}, max_iters = {}'.format(C, max_iters))
        plt.show()
        '''
        return (all_errors, all_fp_violations, all_fn_violations)

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
                        fairness_def='FP',
                        predictor=linear_model.LinearRegression()):
        self.C = C
        self.printflag = printflag
        self.heatmapflag = heatmapflag
        self.heatmap_iter = heatmap_iter
        self.heatmap_path = heatmap_path
        self.max_iters = max_iters
        self.gamma = gamma
        self.fairness_def = fairness_def
        self.predictor = predictor
        if self.fairness_def != 'FP':
            raise Exception('This metric is not yet supported for learning. Metric specified: {}.'.format(self.fairness_def))


