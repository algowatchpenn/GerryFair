# to do: discard some columns to enlarge dataset
import sys
import numpy as np
import pandas as pd
from sklearn import linear_model
import random
from Reg_Oracle_Fict import *
random.seed(1)
num_sens, dataset = sys.argv[1:]
num_sens = int(num_sens)
dataset = str(dataset)
random.seed(1)

# print out the invoked parameters
print('Invoked Parameters: number of sensitive attributes = {}, dataset = {}'.format(num_sens, dataset))

# Data Cleaning and Import
f_name = 'clean_{}'.format(dataset)
clean_the_dataset = getattr(clean_data, f_name)
X, X_prime, y = clean_the_dataset(num_sens)

def get_fp(preds, y):
    return np.mean([p for i,p in enumerate(preds) if y[i] == 0])


def audit(predictions, X, X_prime, y):

    FP = get_fp(predictions, y)
    aud_group, gamma_unfair, fp_in_group, err_group = get_group(predictions, X_sens=X_prime, X=X, y_g=y, FP=FP)
    group_size = gamma_unfair/fp_in_group
    group_coefs = aud_group.b0.coef_ - aud_group.b1.coef_
    print('group size: {}, gamma-unfairness: {}, FP-disparity: {}'.format(group_size, gamma_unfair, fp_in_group))
    print('subgroup_coefficients: {}'.format(group_coefs),)
    print('sensitive attributes: {}'.format([c for c in X_prime.columns],))

if __name__ == "__main__":
    pass




