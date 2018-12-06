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
    # Output: for each iteration, the error and the fp difference - heatmap can also be produced
    def _fictitious_play(self,
                        X,
                        X_prime,
                        y):

        # defining variables and data structures for algorithm
        learner = Learner(X, y)
        auditor = Auditor()

        n = X.shape[0]
        m = len([s for s in y if s == 0])
        p = [learner.best_response([1.0 / n] * m)]
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

        # scaling variables for heatmap
        vmin = None
        vmax = None

        while iteration < self.max_iters:
            print('iteration: {}'.format(int(iteration)))
            # get t-1 mixture decisions on X by randomizing on current set of p
            emp_p = learner.generate_predictions(p[-1], A, iteration)
            # get the error of the t-1 mixture classifier
            err = emp_p[0]
            # Average decisions
            A = emp_p[1]

            # update FP to get the false positive rate of the mixture classifier
            A_recent = p[-1].predict(X)
            # FP rate of t-1 mixture on new group g_t
            FP_recent = np.mean([A_recent[i] for i, c in enumerate(y) if c == 0])
            FP = ((iteration - 1.0) / iteration) * FP + FP_recent * (1.0 / iteration)
            
            # dual player best responds to strategy up to t-1
            f = auditor.get_group(A, X_prime, y, FP)
            # flag whether FP disparity was positive or negative
            pos_neg = f[4]
            fp_disparity = f[1]
            group_size_0 = np.sum(f[0].predict(X_0)) * (1.0 / n)

            # compute list of people who have been included in an identified subgroup up to time t
            group_membership = np.add(group_membership, f[0].predict(X_prime))
            group_membership = [g != 0 for g in group_membership]
            group_members_t = np.sum(group_membership)
            cum_group_mems.append(group_members_t)

            # compute learner's best response to the CSC problem
            p_t = learner.best_response(c_1t)
            A_t = p_t.predict(X)
            FP_t = np.mean([A_t[i] for i, c in enumerate(y) if c == 0])

            # append new group, new p, fp_diff of group found, coefficients, group size
            groups.append(f[0])
            p.append(p_t)
            fp_diff_t.append(np.abs(f[1]))
            errors_t.append(err)
            coef_t.append(f[0].b0.coef_ - f[0].b1.coef_)


            # outputs
            if iteration == 1:
                print(
                    'most accurate classifier accuracy: {}, most acc-class unfairness: {}, most acc-class size {}'.format(
                        err,
                        fp_diff_t[0],
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
            c_1t = auditor.update_costs(c_1t, f, X_prime, y, self.C, iteration, fp_disparity, self.gamma)
            iteration += 1    

        self.classifiers = p
        return errors_t, fp_diff_t

    def predict(self, X):
        num_classifiers = len(self.classifiers)

        y_hat = None
        for c in self.classifiers: 
            new_preds = np.multiply(1.0 / num_classifiers, c.predict(X))
            if y_hat is None:
                y_hat = new_preds
            else:
                y_hat = np.add(y_hat, new_preds)
        return pd.DataFrame(y_hat)

    def pareto(self, X, X_prime, y, gamma_list, C=10, max_iters=10):
        # Store errors and fp over time for each gamma
        all_errors = []
        all_fp = []
        self.C = C
        self.max_iters = max_iters
        for g in gamma_list:
            self.gamma = g
            errors_gt, fp_diff_gt = self._fictitious_play(X, X_prime, y)
            print(errors_gt, fp_diff_gt)
            all_errors.append(np.mean(errors_gt))
            all_fp.append(np.mean(fp_diff_gt))
        plt.plot(all_errors, all_fp)
        plt.xlabel('error')
        plt.ylabel('unfairness (fp_diff*size)')
        plt.title('error vs. unfairness: C = {}, max_iters = {}'.format(C, max_iters))
        plt.show()
        return (all_errors, all_fp)

    def train(self, X, X_prime, y, alg="fict"):
        if alg == "fict":
            err, fp_diff = self._fictitious_play(X, X_prime, y)
            return err, fp_diff
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
                        gamma=None):
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

    def __init__(self, C=10,
                        printflag=False,
                        heatmapflag=False,
                        heatmap_iter=10,
                        heatmap_path='.',
                        max_iters=10,
                        gamma=0.01):
        self.C = C
        self.printflag = printflag
        self.heatmapflag = heatmapflag
        self.heatmap_iter = heatmap_iter
        self.heatmap_path = heatmap_path
        self.max_iters = max_iters
        self.gamma = gamma

class Learner:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def best_response(self, c_1t):
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
    def generate_predictions(self, q, A, iteration):
        """Return the classifications of the average classifier at time iter."""

        new_preds = np.multiply(1.0 / iteration, q.predict(self.X))
        ds = np.multiply((iteration - 1.0) / iteration, A)
        ds = np.add(ds, new_preds)
        error = np.mean([np.abs(ds[k] - self.y[k]) for k in range(len(self.y))])
        return [error, ds]

class Auditor:
    """docstring for Auditor"""
    def update_costs(self, c_1, f, X_prime, y, C, iteration, fp_disp, gamma):
        """Recursively update the costs from incorrectly predicting 1 for the learner."""
        # store whether FP disparity was + or -
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
        """
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values
        FP = np.mean([p for i,p in enumerate(predictions) if y[i] == 0])
        aud_group, gamma_unfair, fp_in_group, err_group, pos_neg = self.get_group(predictions, X_sens=X_prime, y_g=y, FP=FP)

        return aud_group.predict(X_prime), gamma_unfair
