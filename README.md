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
python Reg_Oracle_Fict.py C num_sensitive_features printflag dataset reg_oracle max_iterations gamma_unfairness 'gamma'
```
To audit for gamma unfairness on a dataset:
```python
python Audit.py num_sensitive_feautures dataset max_iterations 
```

## License
Universit of Pennsylvania, Microsoft Research, Seth Neel, Michael Kearns, Aaron Roth, Steven Wu.

## Acknowledgments

* Thank you to the authors of: http://fatml.mysociety.org/media/documents/reductions_approach_to_fair_classification.pdf   , whose algorithm is implemented in MSR_Reduction.py and evaluated in Audit.py
