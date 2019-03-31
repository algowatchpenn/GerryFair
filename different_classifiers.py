import gerryfair
from gerryfair.model import *
from gerryfair.clean import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn import svm
from sklearn import tree
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model

import pickle


dataset = "./dataset/communities.csv"
attributes = "./dataset/communities_protected.csv"
centered = True
X, X_prime, y = clean_dataset(dataset, attributes, centered)

# 3-layer relu network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_hidden1 = 32
        num_hidden2 = 8
        self.lin1 = nn.Linear(X.shape[1], num_hidden1)
        self.lin2 = nn.Linear(num_hidden1, num_hidden2)
        self.lin3 = nn.Linear(num_hidden2, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = self.lin3(x)
        return x


def single_trial():
    ln_predictor = linear_model.LinearRegression()

    svm_predictor = svm.LinearSVR()

    tree_predictor = tree.DecisionTreeRegressor()

    kernel_predictor = KernelRidge(alpha=1.0, gamma=1.0, kernel='rbf')


    C = 15
    printflag = True
    gamma = 0.1
    max_iter = 10
    fair_clf = Model(C=C, printflag=printflag, gamma=gamma, predictor=kernel_predictor, max_iters=max_iter)

    fair_clf.train(X, X_prime, y)




def multiple_comparision():


    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=.5)
    nn_predictor = TorchPredictor(net, criterion, optimizer, 500, initialization=constant_init, device=False)

    ln_predictor = linear_model.LinearRegression()

    svm_predictor = svm.LinearSVR()

    tree_predictor = tree.DecisionTreeRegressor()

    kernel_predictor = KernelRidge(alpha=1.0, gamma=1.0, kernel='rbf')

    C = 15
    printflag = False
    predictor_dict = {'Linear': ln_predictor, 'SVR': svm_predictor,
                      'DT': tree_predictor, 'RBF Kernel': kernel_predictor}
    gamma_list = [0.01]
    max_iter_list = [5]
    results_dict = {}
    # For each model, train a Pareto curve
    for max_iter in max_iter_list:
        for curr_predictor in predictor_dict.keys():
            print('Curr Predictor: ')
            print(curr_predictor)
            predictor = predictor_dict[curr_predictor]
            fair_clf = Model(C=C, printflag=printflag, gamma=1, predictor=predictor, max_iters=max_iter)
            print(fair_clf.predictor)
            all_errors, all_fp = fair_clf.pareto(X, X_prime, y, gamma_list)
            results_dict[curr_predictor] = {'Errors' : all_errors, 'FP_disp': all_fp}

    print(results_dict)

    pickle.dump(results_dict, open('results_max' + str(max_iters) + '_gammas' + str(gamma_list) + '.pkl', 'wb'))


single_trial()