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


class mu1_layer(nn.Module):  # simple model
    """
    $\mu = 1 + \beta_1*\sigmoid(\beta_2 ||x - w||_1 +\beta_3)$
    """

    def __init__(self, num_series):
        super().__init__()
        self.mu1 = nn.Parameter(torch.zeros(num_series, dtype=torch.float64))

    def forward(self):
        return self.mu1


class mu2_layer(nn.Module):  # simple model
    """
    $\mu = 1 + \beta_1*\sigmoid(\beta_2 ||x - w||_1 +\beta_3)$
    """

    def __init__(self, num_series, NumScen):
        super().__init__()
        self.mu2 = nn.Parameter(-0.05 + 0.1 * torch.rand((NumScen, num_series), dtype=torch.float64))
        # nn.Parameter(torch.normal(torch.rand((NumScen, num_series), dtype = torch.float64), std = torch.tensor(1.0)))

    def forward(self):
        return self.mu2


class RegressorModule(nn.Module):
    """
    Implements the neural architecture that maps surrogate scenarios
    and an observed scenario to a value of the predicted loss

    i.e. $\zeta_{1...K}$ and $\xi$ are inputs that are stacked
    Each surrogate scenario is encoded using the encoder $\Phi_1$
    i.e. $\Phi_1(\zeta^{(k)}$
    then each encoded scenario is aggregated as a representative scenario
    and fed into a network $\Phi_2$ used to predict $l(\zeta_{1...K}, \xi)$
    where $l(\zeta_{1...K}, \xi)$ the loss of the surrogate scenario solution
    $\boldsymbol{y}(\zeta_{1...K})$ evaluated on scenario $\xi
    """

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
                nn.Linear(2 * input_dim, num_units),
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
                nn.Linear(2 * input_dim, num_units),
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
                nn.Linear(2 * input_dim, num_units),
                nonlin(),
                nn.Linear(num_units, 1)  # to predict the loss
            )

    def forward(self, X, **kwargs):
        # operate on each scenario
        # aggregate them
        # operate on the aggregated scenario
        lower_X = X[..., :-1, :]  # lower level scenarios
        upper_X = X[..., -1, :]  # upper level scenario
        if self.print_flag:
            print("lower scenarios shape ", lower_X.shape)
            print("upper scenarios shape ", upper_X.shape)

        Phi1 = self.encoder(lower_X)  # encode the lower level
        Agg = Phi1.mean(axis=-2)  # aggregate them
        Agg_upperX = torch.concat([Agg, upper_X], dim=-1)
        X = self.predictor(Agg_upperX)
        return X


class DCSRO_task_net(nn.Module):
    def __init__(
            self,
            input_dim=2 * 50,
            num_series=50,
            num_surrogates=3,
            num_units=3 * 128,  # number of hidden units
            nonlin=nn.ReLU,
            print_flag=False,
            number_hidden=3  # 4, 3, 2

    ):
        super(DCSRO_task_net, self).__init__()
        self.num_series = num_series
        self.num_surrogates = num_surrogates
        # self.mu1 = nn.Parameter(torch.rand((1, num_series), dtype = torch.float64))
        # self.mu2 = nn.Parameter(torch.rand((num_surrogates, num_series), dtype = torch.float64))
        self.num_units = num_units
        self.input_dim = input_dim
        self.nonlin = nonlin
        self.print_flag = print_flag
        self.number_hidden = number_hidden

        if number_hidden >= 4:
            self.mu1_predictor = nn.Sequential(
                nn.Linear(input_dim, num_units),
                nonlin(),
                nn.Linear(num_units, num_units),
                nonlin(),
                nn.Linear(num_units, num_units),
                nonlin(),
                nn.Linear(num_units, self.num_series)
            )

            self.mu2_predictor = nn.Sequential(
                nn.Linear(input_dim, num_units),
                nonlin(),
                nn.Linear(num_units, num_units),
                nonlin(),
                nn.Linear(num_units, num_units),
                nonlin(),
                nn.Linear(num_units, self.num_series * self.num_surrogates)  # to predict the loss
            )

        elif number_hidden == 3:
            self.mu1_predictor = nn.Sequential(
                nn.Linear(input_dim, num_units),
                nonlin(),
                nn.Linear(num_units, num_units),
                nonlin(),
                nn.Linear(num_units, self.num_series)
            )

            self.mu2_predictor = nn.Sequential(
                nn.Linear(input_dim, num_units),
                nonlin(),
                nn.Linear(num_units, num_units),
                nonlin(),
                nn.Linear(num_units, self.num_series * self.num_surrogates)  # to predict the loss
            )

        elif number_hidden == 2:
            self.mu1_predictor = nn.Sequential(
                nn.Linear(input_dim, num_units),
                nonlin(),
                nn.Linear(num_units, self.num_series)
            )

            self.mu2_predictor = nn.Sequential(
                nn.Linear(input_dim, num_units),
                nonlin(),
                nn.Linear(num_units, self.num_series * self.num_surrogates)  # to predict the loss
            )

    def forward(self, X, **kwargs):
        # operate on each scenario
        # aggregate them
        # operate on the aggregated scenario
        Batches = X.size(dim=0)
        flatten_context = X.flatten(start_dim=1)  # batch x k n_series where k is AR(k)

        mu1 = self.mu1_predictor(flatten_context)
        mu1 = mu1[:, None]
        mu2 = self.mu2_predictor(flatten_context)

        # repeat the first stage expected returns K_ times
        repeated_mu1 = torch.repeat_interleave(mu1, repeats=self.num_surrogates, dim=1)

        mu2_reshape = mu2.reshape(Batches, self.num_surrogates, self.num_series)

        # concatenate the first and second period scenarios together
        concatenated_mu = torch.cat((repeated_mu1, mu2_reshape), axis=-1)

        # scratch
        # concatenated_mu = concatenated_mu[None, :]
        # concatenated_mu = torch.repeat_interleave(concatenated_mu, repeats = Batches, dim = 0)
        # print(concatenated_mu.shape)
        return concatenated_mu

