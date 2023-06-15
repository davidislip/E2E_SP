import numpy as np
import cvxpy as cp
import pandas as pd


def single_forecast_mpc(mu_1, mu_2, covariance, risk_aversion=10, transaction_penalty=0.05, y_1_val=None):
    n = len(mu_1)
    w_0 = (1 / n) * np.ones(n)
    y_1 = cp.Variable(n)
    y_2 = cp.Variable(n)
    w_1 = cp.Variable(n)
    w_2 = cp.Variable(n)
    constraints = [cp.sum(w_1) == 1, cp.sum(w_2) == 1,
                   w_1 >= 0, w_2 >= 0,
                   y_1 == w_1 - w_0, y_2 == w_2 - w_1]
    if y_1_val is not None:
        constraints.extend([y_1 == y_1_val])

    ret_1 = mu_1.T @ w_1
    ret_2 = mu_2.T @ w_2
    risk_1 = risk_aversion * cp.quad_form(w_1, covariance)
    risk_2 = risk_aversion * cp.quad_form(w_2, covariance)
    transaction_cost = transaction_penalty * cp.norm1(y_1) + transaction_penalty * cp.norm1(y_2)
    adjusted_returns = ret_1 + ret_2 - (risk_1 + risk_2 + transaction_cost)
    prob = cp.Problem(cp.Maximize(adjusted_returns), constraints)
    prob.solve()

    df = pd.DataFrame((adjusted_returns.value, ret_1.value + ret_2.value, \
                       risk_1.value + risk_2.value, transaction_cost.value),
                      index=['Risk and Txn Adjusted Return', 'Return', 'Risk', 'Transaction Cost'])
    return df, y_1.value, y_2.value, w_1.value, w_2.value


def multi_forecast_mpc(pi, mu_1, mu_forecasts, covariance, risk_aversion=10, transaction_penalty=0.05):
    # forecasts will be fed in as a list of vectors
    M = len(pi)  # number of scenarios
    n = len(mu_1)  #

    w_0 = (1 / n) * np.ones(n)  # initial portfolio
    y = cp.Variable(n)
    w = cp.Variable(n)
    y_scenario = cp.Variable((n, M))  # second period variable in scenario
    w_scenario = cp.Variable((n, M))  # second period variable in scenario
    constraints = [cp.sum(w) == 1, w >= 0, y == w - w_0]
    for i in range(M):
        constraints.extend([cp.sum(w_scenario[:, i]) == 1,
                            w_scenario[:, i] >= 0,
                            y_scenario[:, i] == w_scenario[:, i] - w])

    ret_1 = mu_1.T @ w
    risk_1 = risk_aversion * cp.quad_form(w, covariance)
    transaction_cost = transaction_penalty * cp.norm1(y)
    adjusted_returns = ret_1 - (risk_1 + transaction_cost)
    expected_returns = 0

    for i in range(M):
        ret_i = mu_forecasts[i].T @ w_scenario[:, i]
        risk_i = risk_aversion * cp.quad_form(w_scenario[:, i], covariance)
        transaction_cost_i = transaction_penalty * cp.norm1(y_scenario[:, i])
        expected_returns += pi[i] * (ret_i - (risk_i + transaction_cost_i))

    prob = cp.Problem(cp.Maximize(adjusted_returns + expected_returns), constraints)
    prob.solve(verbose = True)

    df = pd.DataFrame((adjusted_returns.value, expected_returns.value),
                      index=['Stage 1 Objective', 'Stage 2 Objective'])
    return df, y.value, w.value  # today information


class OLS:
    """least squares for the simple predict then optimize"""

    def __init__(self, x, mu1, mu2, y):
        self.x = x
        self.mu1 = mu1
        self.mu2 = mu2
        self.y = y
        self.mu_1_coeffs = None
        self.mu_2_coeffs = None

    def fit_mu1(self):
        '''uses OLS to predict mu1'''
        n = len(self.x)
        X = np.vstack([self.x, np.ones(n)]).T
        self.mu_1_coeffs = np.linalg.lstsq(X, self.mu1, rcond=None)[0]
        return self.mu_1_coeffs

    def predict_mu1(self, x):
        return self.mu_1_coeffs[0] * x + self.mu_1_coeffs[1]

    def fit_mu2(self):
        n = len(self.x)
        X = np.vstack([self.mu1, self.y, np.ones(n)]).T
        self.mu_2_coeffs = np.linalg.lstsq(X, self.mu2, rcond=None)[0]
        return self.mu_2_coeffs

    def predict_mu2(self, mu1, y):
        return self.mu_2_coeffs[0] * mu1 + self.mu_2_coeffs[1] * y + self.mu_2_coeffs[2]
