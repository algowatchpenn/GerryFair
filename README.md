# GerryFair: Auditing and Learning for Subgroup Fairness

Fairness Gerrymandering: when "a classifier appears to be fair on each individual
group, but badly violates the fairness constraint on one or more structured subgroups defined
over the protected attributes" (from Kearns et al., https://arxiv.org/abs/1711.05144)

This repository contains python code for:
* learning fair classifiers subject to subgroup fairness constraints (as described in https://arxiv.org/abs/1711.05144)
* auditing classifier predictions for fairness violations
* visualizing tradeoffs between error and fairness metrics
* fairness sensitive datasets for experiments (as used in https://arxiv.org/abs/1808.08166)

Fairness metrics supported for learning and auditing:
* False Positive Rate equality
* False Negative Rate equality

Learner classes supported:
* Any sklearn binary classifier (defaults to LinearRegression)

Group classes supported:
* Linear threshold functions over protected attributes


### Prerequisites

To install the package and prepare for use, run:
```
git clone https://github.com/algowatchPenn/GerryFair.git
pip install -r requirements.txt
```
The current iteration of the package uses the following python packages: pandas, numpy, sklearn, matplotlib
If you already have these installed, you can forgo the requirements step.

## Using our package

For demonstration of the GerryFair API, please see our [jupyter notebook](./GerryFair&#32;Demo.ipynb). Other examples of usage are provided as scripts in the `examples` folder. These should be run from a location which contains the `gerryfair` folder.

## Datasets
#### communities: http://archive.ics.uci.edu/ml/datasets/communities+and+crime
#### lawschool: https://eric.ed.gov/?id=ED469370
#### adult: https://archive.ics.uci.edu/ml/datasets/adult
#### student: https://archive.ics.uci.edu/ml/datasets/student+performance (math grades)


## License
* GerryFair/license.txt
* Maintained by: Seth Neel (sethneel@wharton.upenn.edu), William Brown, Adel Boyarsky, Arnab Sarker, Aaron Hallac. 
* Property of: Michael Kearns, Seth Neel, Aaron Roth, Z. Steven Wu.
* For questions or concerns, contact Algowatch Project (algowatchproject@gmail.com).


