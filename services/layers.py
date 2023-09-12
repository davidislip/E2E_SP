import torch
from scipy.linalg import sqrtm
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import torch.nn as nn

def end_of_loop_mpc_layer(w_0, covariance, risk_aversion, transaction_penalty):
    """
    convex optimization layer for portfolio optimization to evaluate the
    cost of the policy
    """

    n = len(w_0)
    y_1 = cp.Parameter(n)  # first stage decision is fed into optimizer

    mu_1 = cp.Parameter(n)
    mu_2 = cp.Parameter(n)
    covariance_sqrt = torch.from_numpy(sqrtm(covariance))

    y_2 = cp.Variable(n)
    w_1 = cp.Variable(n)
    w_2 = cp.Variable(n)

    constraints = [cp.sum(w_1) == 1, cp.sum(w_2) == 1,
                   w_1 >= -0.0001, w_2 >= -0.0001,
                   y_1 == w_1 - w_0, y_2 == w_2 - w_1]

    ret_1 = mu_1.T @ w_1
    ret_2 = mu_2.T @ w_2

    risk_1 = risk_aversion * cp.sum_squares(covariance_sqrt @ w_1)
    risk_2 = risk_aversion * cp.sum_squares(covariance_sqrt @ w_2)

    transaction_cost = transaction_penalty * cp.norm1(y_1) + transaction_penalty * cp.norm1(y_2)
    adjusted_returns = ret_1 + ret_2 - (risk_1 + risk_2 + transaction_cost)
    prob = cp.Problem(cp.Maximize(adjusted_returns), constraints)
    # inputs and        outputs
    return CvxpyLayer(prob, [y_1, mu_1, mu_2], [w_1, y_2, w_2])
    # the parameters below will be tensors


# evaluate stochastic optimization end to end via linear model and incorrect scenario specification


def multi_forecast_mpc_layer(w_0, M, covariance, risk_aversion, transaction_penalty):
    """
    convex optimization layer for portfolio optimization
    """

    n = len(w_0)
    # forecasts will be fed in as a list of vectors
    # number of scenarios
    y = cp.Variable(n)
    w = cp.Variable(n)
    mu_1 = cp.Parameter(n)
    covariance_sqrt = torch.from_numpy(sqrtm(covariance))
    mu_forecasts = cp.Parameter((M, n))

    y_scenario = cp.Variable((n, M))  # second period variable in scenario
    w_scenario = cp.Variable((n, M))  # second period variable in scenario
    constraints = [cp.sum(w) == 1, w >= 0, y == w - w_0]
    for i in range(M):
        constraints.extend([cp.sum(w_scenario[:, i]) == 1,
                            w_scenario[:, i] >= 0,
                            y_scenario[:, i] == w_scenario[:, i] - w])

    ret_1 = mu_1.T @ w
    risk_1 = risk_aversion * cp.sum_squares(covariance_sqrt @ w)
    transaction_cost = transaction_penalty * cp.norm1(y)
    adjusted_returns = ret_1 - risk_1 - transaction_cost
    expected_returns = 0

    for i in range(M):
        ret_i = mu_forecasts[i].T @ w_scenario[:, i]
        risk_i = risk_aversion * cp.sum_squares(covariance_sqrt @ w_scenario[:, i])
        transaction_cost_i = transaction_penalty * cp.norm1(y_scenario[:, i])
        expected_returns += (1 / M) * ret_i - (1 / M) * risk_i - (1 / M) * transaction_cost_i

    prob = cp.Problem(cp.Maximize(adjusted_returns + expected_returns), constraints)

    return CvxpyLayer(prob, [mu_1, mu_forecasts], [w, y])


class mu1_layer(nn.Module): #simple model
    """
    $\mu = 1 + \beta_1*\sigmoid(\beta_2 ||x - w||_1 +\beta_3)$
    """

    def __init__(self, num_series ):
        super().__init__()
        self.mu1 = nn.Parameter(torch.zeros(num_series,  dtype = torch.float64))

    def forward(self):
        return self.mu1

class mu2_layer(nn.Module): #simple model
    """
    $\mu = 1 + \beta_1*\sigmoid(\beta_2 ||x - w||_1 +\beta_3)$
    """

    def __init__(self, num_series, NumScen):
        super().__init__()
        self.mu2 = nn.Parameter(-0.05 + 0.1*torch.rand((NumScen, num_series),  dtype = torch.float64))
        #nn.Parameter(torch.normal(torch.rand((NumScen, num_series), dtype = torch.float64), std = torch.tensor(1.0)))

    def forward(self):
        return self.mu2