# Rich Subgroup Fairness

This repository contains python code for both 
* learning fair classifiers subject to the fairness definitions in https://arxiv.org/abs/1711.05144
* auditing standard classifiers from sklearn for unfairness
* fairness sensitive datasets for experiments https://arxiv.org/abs/1808.08166

### Prerequisites

python packages: pandas, numpy, sklearn, matplotlib

## Running the tests

To learn a fair classifier on a dataset in the dataset folder subject to gamma unfairness, use Reg_Oracle_Fict.py.
The usage can be found by typing:

```
python Reg_Oracle_Fict.py -h

    usage: Reg_Oracle_Fict.py [-h] [-C C] [-p] [--heatmap]
                              [--heatmap_iters HEATMAP_ITERS] [-d DATASET]
                              [-i ITERS] [--gamma_unfair GAMMA_UNFAIR] [--plots]

    Reg_Oracle_Fict input parser

    optional arguments:
      -h, --help            show this help message and exit
      -C C                  C is the bound on the maxL1 norm of the dual
                            variables, (Default = 10)
      -p, --print_output    Include this flag to determine whether output is
                            printed, (Default = False)
      --heatmap             Include this flag to determine whether heatmaps are
                            generated, (Default = False)
      --heatmap_iters HEATMAP_ITERS
                            number of iterations heatmap data is saved after,
                            (Default = 1)
      -d DATASET, --dataset DATASET
                            name of the dataset (communities, lawschool, adult,
                            student), (Required)
      -i ITERS, --iters ITERS
                            number of iterations to terminate after, (Default =
                            10)
      --gamma_unfair GAMMA_UNFAIR
                            approximate gamma disparity allowed in subgroups,
                            (Default = .01)
      --plots               Include this flag to determine whether plots of error
                            and unfairness are generated, (Default = False)
```
An example of this usage is:
```
python Reg_Oracle_Fict.py -C 10 -p -h --heatmap_iters 1 -d communities -i 10 -g .01
```
Again, the arguments are:
* -C: bound on the max L1 norm of the dual variables with a default value of 10
* --print_output, -p: flag True or False determines whether output is printed with a default value of False
* --heatmap: flag True or False determines whether heatmaps are generated with a default value of False
* --heatmap_iters:  number of iterations heatmap data is saved after with a default value of 1
* --dataset, -d: name of the dataset (communities, lawschool, adult, student), this is required.
* --iters, -i: number of iterations to terminate after with a default value of 10
* --gamma_unfair, -g: approximate gamma disparity allowed in subgroups with a default value of .01
* --plots: flag True or False determines whether plots are generated with a default value of False

outputs (if ```--print_output``` is included), at each iteration print:
* ave_error: the error of the current mixture of classifiers found by the Learner)
* gamma-unfairness: the gamma disparity witnessed by the subgroup found at the current round by the Auditor
* group_size: the size of the above group conditioned on `y = 0`
* frac included ppl: the fraction of the dataset that has been included in a group found by the Auditor thus far (on `y =0`)
* coefficients of g_t: the coefficients of the hyperplane that defines the group found by the Auditor
* if ```--heatmap``` is included.

To audit for gamma unfairness on a dataset, use Audit.py. The usage can be found by typing:
```
python Audit.py -h

    usage: Audit.py [-h] [-d DATASET] [-i ITERS]

    Audit.py input parser

    optional arguments:
      -h, --help            show this help message and exit
      -d DATASET, --dataset DATASET
                            name of the dataset (communities, lawschool, adult,
                            student, all), (Required)
      -i ITERS, --iters ITERS
                            number of iterations to terminate after, (Default =
                            10)
```

* audits trained logistic regression, SVM, nearest-neighbor model. 
## Datasets
#### communities: http://archive.ics.uci.edu/ml/datasets/communities+and+crime
#### lawschool: https://eric.ed.gov/?id=ED469370
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
* Maintained by: Seth Neel (sethneel@wharton.upenn.edu)
* Property of: Michael Kearns, Seth Neel, Aaron Roth, Z. Steven Wu.

## Acknowledgments

* Thank you to the authors of: http://fatml.mysociety.org/media/documents/reductions_approach_to_fair_classification.pdf, whose algorithm/code is in the `fairlearn` folder, and is audited in `Audit.py`.
