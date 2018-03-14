import numpy as np
import pandas as pd

# documentation
# output: clean data set, remove missing values and convert categorical values to binary, extract sensitive features
# for each data set 'name.csv' we create a function clean_name
# clean name takes parameter num_sens, which is the number of sensitive attributes to include
# clean_name returns pandas data frames X, X_prime, where:
# X is the full data set of X values
# X_prime is only the sensitive columns of X
# y are the binary outcomes


# num_sens in 1:18
def clean_communities(num_sens):
    """Clean communities & crime data set."""
    # Data Cleaning and Import
    df = pd.read_csv('dataset/communities.csv')
    df = df.fillna(0)

    # sensitive variables are just racial distributions in the population and police force as well as foreign status
    # median income and pct of illegal immigrants / related variables are not labeled sensitive
    sens_features = [3, 4, 5, 6, 22, 23, 24, 25, 26, 27, 61, 62, 92, 105, 106, 107, 108, 109]
    df_sens = df.iloc[:, sens_features[0:num_sens]]
    y = df['ViolentCrimesPerPop']
    q_y = np.percentile(y, 70)
    # convert y's to binary predictions on whether the neighborhood is
    # especially violent
    y = [np.round((1 + np.sign(s - q_y)) / 2) for s in y]
    X = df.iloc[:, 0:122]
    X_prime = df_sens
    return X, X_prime, y


# num_sens in 1:17
def clean_lawschool(num_sens):
    """Clean law school data set."""
    # Data Cleaning and Import
    df = pd.read_csv('dataset/lawschool.csv')
    df = df.dropna()
    # convert categorical column variables to 0,1
    df['gender'] = df['gender'].map({'female': 1, 'male': 0})
    # remove y from df
    df_y = df['bar1']
    df = df.drop('bar1', 1)
    y = [int(a == 'P') for a in df_y]
    # one hot coding of race variable
    for i in range(1,9):
        col_name = 'race{}'.format(i)
        race_code = [np.int(r == i) for r in df['race']]
        df[col_name] = race_code
    df = df.drop('race', 1)
    # sensitive variables are just racial distributions in the population and police force as well as foreign status
    # median income and pct of illegal immigrants / related variables are not labeled sensitive
    sens_features = range(df.shape[1])
    x_prime = df.iloc[:, sens_features[0:num_sens]]
    return df, x_prime, y


def clean_synthetic(num_sens):
    """Clean synthetic data set, all features sensitive, y value is last col."""
    df = pd.read_csv('dataset/synthetic.csv')
    df = df.dropna()
    y_col = df.shape[1]-1
    y = df.iloc[:,y_col]
    df = df.iloc[:, 0:(y_col-1)]
    x_prime = df.iloc[:, 0:num_sens]
    return df, x_prime, y




