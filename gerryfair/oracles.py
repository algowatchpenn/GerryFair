import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

# helper function for torch support
def xavier_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def constant_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.constant(m.weight)
        m.bias.data.fill_(0.01)

# 3-layer relu network
class ThreeLayerNet(nn.Module):
    def __init__(self, inputs):
        super(ThreeLayerNet, self).__init__()
        num_hidden1 = 32
        num_hidden2 = 8
        self.lin1 = nn.Linear(inputs, num_hidden1)
        self.lin2 = nn.Linear(num_hidden1, num_hidden2)
        self.lin3 = nn.Linear(num_hidden2, 1)

    def forward(self, x):
        x = func.relu(self.lin1(x))
        x = self.lin2(x)
        x = self.lin3(x)
        return x

class TorchPredictor:
    """Takes in a neural network defined in torch and outputs a valid Predictor"""

    def __init__(self,
                 nn, 
                 criterion,
                 optimizer,
                 batch_size=500,
                 initialization=xavier_init,
                 device=False):
        self.nn = nn
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        self.initialization = initialization

    def fit(self, X, costs):

        #X_tensor = torch.from_numpy(X.values).float()
        costs_tensor = torch.tensor(costs).float()
        print(costs_tensor)

        dataset = Intersectional_Dataset(X, costs)
        # define the dataloader to iterate through dataset
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        num_epochs = 1000
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            for i, data in enumerate(dataloader, 0):
                # get the inputs
                x = data['x'].float()
                y = data['y'].float()
                if self.device:
                    x, y = x.to(device), y.to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.nn(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                # print statistics
                if i == 0 and (epoch % (num_epochs / 10) == 0):
                    print('[%d, %5d] loss: %.12f' %
                          (epoch, i, loss.item()))

        print('Finished Training')
        return self

        # num_iter = 5000
        # for epoch in range(0, num_iter + 1):  # loop over the dataset multiple times
        #
        #     # zero the parameter gradients
        #     self.optimizer.zero_grad()
        #
        #     # forward + backward + optimize
        #     outputs = self.nn(X_tensor)
        #     loss = self.criterion(outputs, costs_tensor)
        #     loss.backward()
        #     self.optimizer.step()
        #
        #     if epoch % (num_iter / 10) == 0:  # print every 100 mini-batches
        #         print('[%d] loss: %.12f' %
        #               (epoch, loss.item()))
        #
        # print('Finished Training')
        # return self

    def predict(self, x_series):
        x_tensor = torch.from_numpy(x_series).float()
        return self.nn(x_tensor).detach().numpy()