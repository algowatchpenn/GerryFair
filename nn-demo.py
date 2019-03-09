import gerryfair
from gerryfair.model import *
from gerryfair.clean import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


dataset = "./dataset/communities.csv"
attributes = "./dataset/communities_protected.csv"
centered = True
X, X_prime, y = clean_dataset(dataset, attributes, centered)

# 3-layer relu network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(122, 32)
        self.lin2 = nn.Linear(32, 8)
        self.lin3 = nn.Linear(8, 1)


    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=.001)
nn_predictor = TorchPredictor(net, criterion, optimizer, 32, initialization=xavier_init, device=False)

C = 15
printflag = True
gamma = .01
fair_model = Model(C=C, printflag=printflag, gamma=gamma, predictor=nn_predictor)
max_iters = 10
fair_model.set_options(max_iters=max_iters)

# Train Set
X_train = X.iloc[:X.shape[0]-50]
X_prime_train = X_prime.iloc[:X_prime.shape[0]-50]
y_train = y.iloc[:y.shape[0]-50]
# Test Set
X_test = X.iloc[-50:].reset_index(drop=True)
X_prime_test = X_prime.iloc[-50:].reset_index(drop=True)
y_test = y.iloc[-50:].reset_index(drop=True)

# Train the model
[errors, fp_difference] = fair_model.train(X_train, X_prime_train, y_train)

predictions = fair_model.predict(X_test)