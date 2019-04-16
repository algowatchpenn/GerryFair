import numpy as np
import pandas as pd
from sklearn import linear_model
from gerryfair.reg_oracle_class import RegOracle

class Group(object):
    """docstring for Group"""

    def return_f(self):
        return [self.func, self.weighted_disparity, self.disparity, self.group_err, self.disparity_direction]

    def __init__(self, func, group_size, weighted_disparity, disparity, group_err, disparity_direction):
        super(Group, self).__init__()
        self.func = func
        self.group_size = group_size
        self.weighted_disparity = weighted_disparity
        self.disparity = disparity
        self.group_err = group_err
        self.disparity_direction = disparity_direction

class Auditor:
    """This is the Auditor class. It is used in the training algorithm to repeatedly find subgroups that break the
    fairness disparity constraint. You can also use it independently as a stand alone auditor."""
    def __init__(self, X_prime, y, fairness_def):
        self.X_prime = X_prime
        self.y_input = y
        self.y_inverse = np.array([abs(1-y_value) for y_value in self.y_input])
        self.fairness_def = fairness_def
        if self.fairness_def not in ['FP', 'FN']:
            raise Exception('Invalid fairness metric specified: {}. Please choose \'FP\' or \'FN\'.'.format(self.fairness_def))
        self.y = self.y_input
        if self.fairness_def == 'FN':
            self.y = self.y_inverse
        self.X_prime_0 = pd.DataFrame([self.X_prime.iloc[u, :] for u, s in enumerate(self.y) if s == 0])
        self.X_prime_1 = pd.DataFrame([self.X_prime.iloc[u, :] for u, s in enumerate(self.y) if s == 0])

    def initialize_costs(self, n):
        costs_0 = None
        costs_1 = None
        if self.fairness_def == 'FP':
            costs_0 = [0.0] * n
            costs_1 = [-1.0 / n * (2 * i - 1) for i in self.y_input]
            
        elif self.fairness_def == 'FN':
            costs_0 = [0.0] * n
            costs_1 = [1.0 / n * (2 * i - 1) for i in self.y_input]
            #X_0 = pd.DataFrame([X_prime.iloc[u, :] for u, s in enumerate(y) if s == 1])
        return costs_0, costs_1, self.X_prime_0

    def get_baseline(self, y, y_hat):
        if self.fairness_def == 'FP':
            return np.mean([y_hat[i] for i, c in enumerate(y) if c == 0])
        elif self.fairness_def == 'FN':
            return np.mean([(1 - y_hat[i]) for i, c in enumerate(y) if c == 1])

    def update_costs(self, c_0, c_1, group, C, iteration, gamma):
        """Recursively update the costs from incorrectly predicting 1 for the learner."""
        # store whether FP disparity was + or -
        f = group.return_f()
        pos_neg = f[4]
        n = len(self.y)

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

        return c_0, c_1

    def get_subset(self, predictions):
        if self.fairness_def == 'FP':
            return self.X_prime_0, [a for u, a in enumerate(predictions) if self.y[u] == 0]
        elif self.fairness_def == 'FN':
            return self.X_prime_1, [a for u, a in enumerate(predictions) if self.y[u] == 0] # changed 1 to 0, inverting labels

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
        if (fp_disp_w_neg > fp_disp_w):
            return Group(func_neg, g_size_0_neg, fp_disp_w_neg, fp_disp_neg, err_group_neg, -1)
        else:
            return Group(func, g_size_0, fp_disp_w, fp_disp, err_group, 1)

    def audit(self, predictions):
        """Takes in predictions on dataset (X',y) and returns:
            a vector which represents the group that violates the fairness metric, along with the u.
        """
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values

        metric_baseline = self.get_baseline(self.y_input, predictions)
        aud_group, gamma_unfair, fp_in_group, err_group, pos_neg = self.get_group(predictions, metric_baseline)

        return aud_group.predict(self.X_prime), gamma_unfair
