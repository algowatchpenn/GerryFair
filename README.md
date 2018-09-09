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
### communities: http://archive.ics.uci.edu/ml/datasets/communities+and+crime
### lawschool: 
### adult: https://archive.ics.uci.edu/ml/datasets/adult
### student: https://archive.ics.uci.edu/ml/datasets/student+performance (math grades)
### synthetic dataset

### Adding a dataset
Let's add a dataset `credit_scores`. The actual dataset
should be saved in a standard csv format where rows correspond to observations, and with the columns named.
This file should be called `dataset/credit_scores.csv`. These columns can include the target variable `y`. The second file, should be saved as `dataset/credit_scores_protected.csv`, and should have the same column names as `credit_scores.csv` except for the target variable, and only one row. Each entry should be 0 or 1, depending on whether that column is designated as a protected attribute. After these csv files are saved in the dataset folder, we create a function in `clean_data.py` that preprocesses the dataset to be fed into the algorithm in `Reg_Oracle_Fict.py`.

## License
Seth Neel, Michael Kearns, Aaron Roth, Steven Wu.

## Acknowledgments

* Thank you to the authors of: http://fatml.mysociety.org/media/documents/reductions_approach_to_fair_classification.pdf, whose algorithm is implemented in [MSR_Reduction.py](MSR_Reduction.py) and evaluated in [Audit.py](Audit.py)
