import numpy as np
import cvxpy as cp
import pandas as pd
import time 

def single_forecast_mpc(mu_1, mu_2, covariance, risk_aversion, transaction_penalty, y_1_val=None):
    n = len(mu_1)
    w_0 = (1 / n) * np.ones(n)
    y_1 = cp.Variable(n)
    y_2 = cp.Variable(n)
    w_1 = cp.Variable(n)
    w_2 = cp.Variable(n)
    if y_1_val is None:
        constraints = [cp.sum(w_1) == 1, cp.sum(w_2) == 1,
                       w_1 >= 0, w_2 >= 0,
                       y_1 == w_1 - w_0, y_2 == w_2 - w_1]
    else:
        constraints = [cp.sum(w_1) == 1, cp.sum(w_2) == 1,
                       w_1 >= -10**(-4), w_2 >= -10**(-4),
                       y_1 == w_1 - w_0, y_2 == w_2 - w_1,
                       y_1 == y_1_val]

    ret_1 = mu_1.T @ w_1
    ret_2 = mu_2.T @ w_2
    risk_1 = risk_aversion * cp.quad_form(w_1, covariance)
    risk_2 = risk_aversion * cp.quad_form(w_2, covariance)
    transaction_cost = transaction_penalty * cp.norm1(y_1) + transaction_penalty * cp.norm1(y_2)
    adjusted_returns = ret_1 + ret_2 - (risk_1 + risk_2 + transaction_cost)
    prob = cp.Problem(cp.Maximize(adjusted_returns), constraints)
    prob.solve(verbose=False, solver = 'ECOS')

    df = pd.DataFrame((adjusted_returns.value, ret_1.value + ret_2.value, \
                       risk_1.value + risk_2.value, transaction_cost.value),
                      index=['Risk and Txn Adjusted Return', 'Return', 'Risk', 'Transaction Cost'])
    return df, y_1.value, y_2.value, w_1.value, w_2.value


#extensive form of multi forecast problem
def multi_forecast_mpc(pi, mu_1, mu_forecasts, covariance, risk_aversion, transaction_penalty):
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
    prob.solve(verbose = False, solver = 'ECOS')

    df = pd.DataFrame((adjusted_returns.value, expected_returns.value),
                      index=['Stage 1 Objective', 'Stage 2 Objective'])
    return df, y.value, w.value  # today information


def evaluate_portfolio_models(mu1, mu2, cov_matrix, transaction_penalty, risk_aversion):

  #evaluate multi forecast
  two_stage_performance = []
  deterministic_performance = []
  M, n = mu2.shape
  predicted_mu1 = mu1.mean(axis = 0)
  #generate mu2 using the best scenarios i.e. we do not know which one will occur
  mu_forecasts = []
  pi = []
  for scenario in range(M):
      predicted_mu2 =  mu2[scenario,:] #simple_estimator.predict_mu2(predicted_mu1, y_scenarios[scenario,:]) #mu2[scenario,:] #simple_estimator.predict_mu2(predicted_mu1, y_scenarios[scenario,:])
      mu_forecasts.append(predicted_mu2)
      pi.append(1/M)

  start = time.time()
  df_mf, y_val, w_val = multi_forecast_mpc(pi, predicted_mu1, mu_forecasts, cov_matrix, risk_aversion, transaction_penalty =transaction_penalty)
  end = time.time()

  df_sf, y_val_sf, y_2_sf, w_1_sf, w_2_sf = single_forecast_mpc(predicted_mu1, mu2.mean(axis = 0), cov_matrix, risk_aversion = risk_aversion, transaction_penalty =transaction_penalty)

  for scenario in range(M):
      df, y_1, y_2, w_1, w_2 = single_forecast_mpc(mu1[scenario,:], mu2[scenario,:], cov_matrix, risk_aversion = risk_aversion, transaction_penalty =transaction_penalty, y_1_val=y_val)
      two_stage_performance.append(df.T)

      df, y_1, y_2, w_1, w_2 = single_forecast_mpc(mu1[scenario,:], mu2[scenario,:], cov_matrix, risk_aversion = risk_aversion, transaction_penalty =transaction_penalty, y_1_val=y_val_sf)
      deterministic_performance.append(df.T)
  two_stage_performance_df = pd.concat(two_stage_performance, axis =0)
  deterministic_performance_df = pd.concat(deterministic_performance, axis =0)

  print("Transaction penalty ", transaction_penalty)
  print("VSS ", (two_stage_performance_df.mean().iloc[0] - deterministic_performance_df.mean().iloc[0]))

  print("MFC Time (s) ", end - start)

  return {'deterministic portfolio': w_1_sf, 'deterministic results': deterministic_performance_df,
          'MFC portfolio':w_val, 'MFC results':two_stage_performance_df}


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
