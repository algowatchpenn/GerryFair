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
                          [-a ATTRIBUTES] [-i ITERS] [-g GAMMA_UNFAIR]
                          [--plots]

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
                            name of the dataset file (Required)
      -a ATTRIBUTES, --attributes ATTRIBUTES
                            name of the file representing which attributes are
                            protected (unprotected = 0, protected = 1, label = 2)
                            (Required)
      -i ITERS, --iters ITERS
                            number of iterations to terminate after, (Default =
                            10)
      -g GAMMA_UNFAIR, --gamma_unfair GAMMA_UNFAIR
                            approximate gamma disparity allowed in subgroups,
                            (Default = .01)
      --plots               Include this flag to determine whether plots of error
                            and unfairness are generated, (Default = False)

```
An example of this usage is:
```
python Reg_Oracle_Fict.py -C 10 -p -h --heatmap_iters 1 -d dataset/communities_formatted.csv -a dataset/communities_protected_formatted.csv -i 10 -g .01
```
Again, the arguments are:
* -C: bound on the max L1 norm of the dual variables with a default value of 10
* --print_output, -p: flag True or False determines whether output is printed with a default value of False
* --heatmap: flag True or False determines whether heatmaps are generated with a default value of False
* --heatmap_iters:  number of iterations heatmap data is saved after with a default value of 1
* --dataset, -d: name of the dataset, this is required.
* --attributes, -a: name of attribute file, this is required.
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


### Testing on a custom dataset
To test on a custom dataset, two files are needed: a file for the dataset itself and a file listing the types of attributes
in the dataset. The dataset itself should use one-hot encoding for categorical variables, and the label column should have
values in 0,1. For the attributes, each column should have a corresponding label, 0 (unprotected attribute), 1 (protected attribute),
or 2 (label). See `communities_protected_formatted.csv` for an example.


## License
* Maintained by: Seth Neel (sethneel@wharton.upenn.edu)
* Property of: Michael Kearns, Seth Neel, Aaron Roth, Z. Steven Wu.

## Acknowledgments

* Thank you to the authors of: http://fatml.mysociety.org/media/documents/reductions_approach_to_fair_classification.pdf, whose algorithm/code is in the `fairlearn` folder, and is audited in `Audit.py`.
