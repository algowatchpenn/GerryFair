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


def center(X):
    for col in X.columns:
        X.loc[:, col] = X.loc[:, col]-np.mean(X.loc[:, col])
    return X


def add_intercept(X):
    """Add all 1's column to predictor matrix"""
    X['intercept'] = [1]*X.shape[0]


def one_hot_code(df1, sens_dict):
    cols = df1.columns
    for c in cols:
        if isinstance(df1[c][0], basestring):
            column = df1[c]
            df1 = df1.drop(c, 1)
            unique_values = list(set(column))
            n = len(unique_values)
            if n > 2:
                for i in range(n):
                    col_name = '{}.{}'.format(c, i)
                    col_i = [1 if el == unique_values[i] else 0 for el in column]
                    df1[col_name] = col_i
                    sens_dict[col_name] = sens_dict[c]
                del sens_dict[c]
            else:
                col_name = c
                col = [1 if el == unique_values[0] else 0 for el in column]
                df1[col_name] = col
    return df1, sens_dict


# center data frame columns for visual purposes
def clean_communities():
    """Clean communities & crime data set."""
    # Data Cleaning and Import
    df = pd.read_csv('dataset/communities.csv')
    df = df.fillna(0)
    y = df['ViolentCrimesPerPop']
    q_y = np.percentile(y, 70)
    # convert y's to binary predictions on whether the neighborhood is
    # especially violent
    y = [np.round((1 + np.sign(s - q_y)) / 2) for s in y]
    X = df.iloc[:, 0:122]
    # hot code categorical variables
    sens_df = pd.read_csv('dataset/communities_protected.csv')
    sens_cols = [str(c) for c in sens_df.columns if sens_df[c][0] == 1]
    print('sensitive features: {}'.format(sens_cols))
    sens_dict = {c: 1 if c in sens_cols else 0 for c in df.columns}
    df, sens_dict = one_hot_code(df, sens_dict)
    sens_names = [key for key in sens_dict.keys() if sens_dict[key] == 1]
    print('there are {} sensitive features including derivative features'.format(len(sens_names)))
    x_prime = df[sens_names]
    #X = center(X)
    # X = add_intercept(X)
    #x_prime = center(x_prime)
    # x_prime = add_intercept(x_prime)
    return X, x_prime, y


# def clean_communities():
#     num_sens = 18
#     # Data Cleaning and Import
#     df = pd.read_csv('dataset/communities.csv')
#     df = df.fillna(0)
#
#     # sensitive variables are just racial distributions in the population and police force as well as foreign status
#     # median income and pct of illegal immigrants / related variables are not labeled sensitive
#     sens_features = [3, 4, 5, 6, 22, 23, 24, 25, 26, 27, 61, 62, 92, 105, 106, 107, 108, 109]
#     df_sens = df.iloc[:, sens_features[0:num_sens]]
#     y = df['ViolentCrimesPerPop']
#     q_y = np.percentile(y, 70)
#     # convert y's to binary predictions on whether the neighborhood is
#     # especially violent
#     y = [np.round((1 + np.sign(s - q_y)) / 2) for s in y]
#     X = df.iloc[:, 0:122]
#     X_prime = df_sens
#     return X, X_prime, y


# num_sens in 1:17
def clean_lawschool():
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
    sens_df = pd.read_csv('dataset/lawschool_protected.csv')
    sens_cols = [str(c) for c in sens_df.columns if sens_df[c][0] == 1]
    sens_dict = {c: 1 if c in sens_cols else 0 for c in df.columns}
    # one hot coding of race variable
    for i in range(1, 9):
        col_name = 'race{}'.format(i)
        if 'race' in sens_cols:
            sens_dict[col_name] = 1
        else:
            sens_dict[col_name] = 0
        race_code = [np.int(r == i) for r in df['race']]
        df[col_name] = race_code
    sens_dict['race'] = 0
    df = df.drop('race', 1)
    sens_names = [key for key in sens_dict.keys() if sens_dict[key] == 1]
    print('there are {} sensitive features including derivative features'.format(len(sens_names)))
    x_prime = df[sens_names]
    return df, x_prime, y


def clean_synthetic(num_sens):
    """Clean synthetic data set, all features sensitive, y value is last col."""
    df = pd.read_csv('dataset/synthetic.csv')
    df = df.dropna()
    y_col = df.shape[1]-1
    y = df.iloc[:, y_col]
    df = df.iloc[:, 0:y_col]
    x_prime = df.iloc[:, 0:num_sens]
    return df, x_prime, y


def clean_adult():
    df = pd.read_csv('dataset/adult.csv')
    df = df.dropna()
    # binarize and remove y value
    df['income'] = df['income'].map({' <=50K': 0, ' >50K': 1})
    y = df['income']
    df = df.drop('income', 1)
    # hot code categorical variables
    sens_df = pd.read_csv('dataset/adult_protected.csv')
    sens_cols = [str(c) for c in sens_df.columns if sens_df[c][0] == 1]
    print('sensitive features: {}'.format(sens_cols))
    sens_dict = {c: 1 if c in sens_cols else 0 for c in df.columns}
    df, sens_dict = one_hot_code(df, sens_dict)
    sens_names = [key for key in sens_dict.keys() if sens_dict[key] == 1]
    print('there are {} sensitive features including derivative features'.format(len(sens_names)))
    x_prime = df[sens_names]
    return df, x_prime, y

def clean_adultshort():
    df = pd.read_csv('dataset/adultshort.csv')
    df = df.dropna()
    # binarize and remove y value
    df['income'] = df['income'].map({' <=50K': 0, ' >50K': 1})
    y = df['income']
    df = df.drop('income', 1)
    # hot code categorical variables
    sens_df = pd.read_csv('dataset/adultshort_protected.csv')
    sens_cols = [str(c) for c in sens_df.columns if sens_df[c][0] == 1]
    print('sensitive features: {}'.format(sens_cols))
    sens_dict = {c: 1 if c in sens_cols else 0 for c in df.columns}
    df, sens_dict = one_hot_code(df, sens_dict)
    sens_names = [key for key in sens_dict.keys() if sens_dict[key] == 1]
    print('there are {} sensitive features including derivative features'.format(len(sens_names)))
    x_prime = df[sens_names]
    x = center(df)
    x_prime = center(x_prime)
    return x, x_prime, y





# currently 6 sensitive attributes
def clean_student():
    df = pd.read_csv('dataset/student/student-mat.csv', sep=';')
    df = df.dropna()
    y = df['G3']
    y = [0 if y < 11 else 1 for y in y]
    df = df.drop(['G3', 'G2', 'G1'], 1)
    sens_df = pd.read_csv('dataset/student/student_protected.csv')
    sens_cols = [str(c) for c in sens_df.columns if sens_df[c][0] == 1]
    print('sensitive features: {}'.format(sens_cols))
    sens_dict = {c: 1 if c in sens_cols else 0 for c in df.columns}
    df, sens_dict = one_hot_code(df, sens_dict)
    sens_names = [key for key in sens_dict.keys() if sens_dict[key] == 1]
    print('there are {} sensitive features including derivative features'.format(len(sens_names)))
    x_prime = df[sens_names]
    return df, x_prime, y









