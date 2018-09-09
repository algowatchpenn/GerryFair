# Fairness Gerrymandering

This repository contains python code for both 
* learning fair classifiers subject to the fairness definitions in https://arxiv.org/abs/1711.05144
* auditing standard classifiers from sklearn for unfairness
* fairness sensitive datasets for experiments

### Prerequisites

python packages: pandas, numpy, sklearn 

## Running the tests

To learn a fair classifier on a dataset in the dataset folder subject to gamma unfairness:
```python
python Reg_Oracle_Fict.py C printflag, heatmapflag, max_iterations gamma_unfairness
```
arguments: 
* C: bound on the max L1 norm of the dual variables
* printflag: flag True or False determines whether output is printed
* heatmapflag: flag True or False determines whether heatmaps are generated 
* dataset: name of the dataset (communities, lawschool, adult, student)
* max_iterations: number of iterations to terminate after
* gamma_unfairness: approximate gamma disparity allowed in subgroups

To audit for gamma unfairness on a dataset:
```python
python Audit.py num_sensitive_feautures dataset max_iterations 
```
## Datasets
#### communities: http://archive.ics.uci.edu/ml/datasets/communities+and+crime
#### lawschool: 
#### adult: https://archive.ics.uci.edu/ml/datasets/adult
#### student: https://archive.ics.uci.edu/ml/datasets/student+performance (math grades)


### Adding a dataset
Let's add a dataset `credit_scores`. The actual dataset
should be saved in a standard csv format where rows correspond to observations, and with the columns named.
This file should be called `dataset/credit_scores.csv`. These columns can include the target variable `y`. The second file, should be saved as `dataset/credit_scores_protected.csv`, and should have the same column names as `credit_scores.csv` except for the target variable, and only one row. Each entry should be 0 or 1, depending on whether that column is designated as a protected attribute. After these csv files are saved in the dataset folder, we create a function in `clean_data.py` that preprocesses the dataset to be fed into the algorithm in `Reg_Oracle_Fict.py`. The function should be called clean_credit_scores(). Here is an example for the communities dataset: 
```python
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
    X = center(X)
    # X = add_intercept(X)
    x_prime = center(x_prime)
    # x_prime = add_intercept(x_prime)
    return X, x_prime, pd.Series(y)
   ```
   The clean_credit_scores() function reads in the two csv files and returns 3 pandas dataframes: X, X', y. 
   X is the dataframe of all attributes, with all categorical variables one-hot encoded, and all missing or NA data removed or imputed. X' is the dataframe 
   of only the sensitive variables, and y is the target variable. The clean function can also perform other pre-processing     such as centering the columns (`center()`) or adding an intercept (`add_intercept`). Once the clean function has been added, the dataset can now be fed in as a command line argument for `Reg_Oracle_Fict.py`.


## License
Seth Neel, Michael Kearns, Aaron Roth, Steven Wu.

## Acknowledgments

* Thank you to the authors of: http://fatml.mysociety.org/media/documents/reductions_approach_to_fair_classification.pdf, whose algorithm is implemented in [MSR_Reduction.py](MSR_Reduction.py) and evaluated in [Audit.py](Audit.py)
