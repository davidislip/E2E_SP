import torch
from scipy.linalg import sqrtm
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import torch.nn as nn


class RegressorModule(nn.Module):
    def __init__(
            self,
            input_dim=2 * 50,
            num_units=128,  # number of hidden units
            nonlin=nn.ReLU,  # activation function
            print_flag=False,
            number_hidden=4  # 4, 3, 2
    ):
        super(RegressorModule, self).__init__()
        self.num_units = num_units
        self.input_dim = input_dim
        self.nonlin = nonlin
        self.print_flag = print_flag
        self.number_hidden = number_hidden

        if number_hidden >= 4:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, num_units),
                nonlin(),
                nn.Linear(num_units, num_units),
                nonlin(),
                nn.Linear(num_units, num_units),
                nonlin(),
                nn.Linear(num_units, input_dim)
            )

            self.predictor = nn.Sequential(
                nn.Linear(input_dim, num_units),
                nonlin(),
                nn.Linear(num_units, num_units),
                nonlin(),
                nn.Linear(num_units, num_units),
                nonlin(),
                nn.Linear(num_units, 1)  # to predict the loss
            )

        elif number_hidden == 3:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, num_units),
                nonlin(),
                nn.Linear(num_units, num_units),
                nonlin(),
                nn.Linear(num_units, input_dim)
            )

            self.predictor = nn.Sequential(
                nn.Linear(input_dim, num_units),
                nonlin(),
                nn.Linear(num_units, num_units),
                nonlin(),
                nn.Linear(num_units, 1)  # to predict the loss
            )

        elif number_hidden == 2:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, num_units),
                nonlin(),
                nn.Linear(num_units, input_dim)
            )

            self.predictor = nn.Sequential(
                nn.Linear(input_dim, num_units),
                nonlin(),
                nn.Linear(num_units, 1)  # to predict the loss
            )

    def forward(self, X, **kwargs):
        # operate on each scenario
        # aggregate them
        # operate on the aggregated scenario
        X = self.encoder(X)
        X = X.mean(axis=-2)
        X = self.predictor(X)
        return X
