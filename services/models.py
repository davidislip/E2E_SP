import numpy as np
import cvxpy as cp
import pandas as pd
import time
from multiprocessing import Pool
import multiprocessing as mp


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


def define_multi_forecast_mpc(pi, covariance, risk_aversion, transaction_penalty):
    """
    function returns a problem instance
    """
    # forecasts will be fed in as a list of vectors
    M = len(pi)  # number of scenarios
    n, _ = covariance.shape  # number of assets

    # parameter definition
    mu_1 = cp.Parameter(n)
    mu_forecasts = cp.Parameter((M, n))

    M = len(pi)  # number of scenarios

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

    prob._mu1 = mu_1
    prob._mu_forecasts = mu_forecasts
    prob._adjusted_returns = adjusted_returns
    prob._expected_returns = expected_returns
    prob._y = y
    prob._w = w
    return prob  # today information


def define_single_forecast_mpc(covariance, risk_aversion, transaction_penalty, y_1_val_flag=False):
    """
    function that returns an instance of single scenario multi period optimization
    """
    n, n = covariance.shape  #
    w_0 = (1 / n) * np.ones(n)
    # parameter definition
    mu_1 = cp.Parameter(n)
    mu_2 = cp.Parameter(n)

    y_1 = cp.Variable(n)
    y_2 = cp.Variable(n)
    w_1 = cp.Variable(n)
    w_2 = cp.Variable(n)

    if y_1_val_flag is False:
        constraints = [cp.sum(w_1) == 1, cp.sum(w_2) == 1,
                       w_1 >= 0, w_2 >= 0,
                       y_1 == w_1 - w_0, y_2 == w_2 - w_1]
    else:
        y_1_val = cp.Parameter(n)
        constraints = [cp.sum(w_1) == 1, cp.sum(w_2) == 1,
                       w_1 >= -10 ** (-4), w_2 >= -10 ** (-4),
                       y_1 == w_1 - w_0, y_2 == w_2 - w_1,
                       y_1 == y_1_val]

    ret_1 = mu_1.T @ w_1
    ret_2 = mu_2.T @ w_2
    risk_1 = risk_aversion * cp.quad_form(w_1, covariance)
    risk_2 = risk_aversion * cp.quad_form(w_2, covariance)
    transaction_cost = transaction_penalty * cp.norm1(y_1) + transaction_penalty * cp.norm1(y_2)
    adjusted_returns = ret_1 + ret_2 - (risk_1 + risk_2 + transaction_cost)
    prob = cp.Problem(cp.Maximize(adjusted_returns), constraints)

    prob._adjusted_returns = adjusted_returns
    prob._ret_1 = ret_1
    prob._ret_2 = ret_2
    prob._risk_1 = risk_1
    prob._risk_2 = risk_2
    prob._transaction_cost = transaction_cost
    prob._y_1 = y_1
    prob._y_2 = y_2
    prob._w_1 = w_1
    prob._w_2 = w_2
    # add params
    if y_1_val_flag is True:
        prob._y_1_val = y_1_val
    prob._mu_1 = mu_1
    prob._mu_2 = mu_2

    return prob


def solve_single_forecast_mpc(prob, mu_1, mu_2, y_1_val=None):
    """
    prob: cvxpy problem intance
    mu1 numpy array vector of expected first period returns
    mu2 numpy array vector of expected second period returns
    """
    prob._mu_1.value = mu_1
    prob._mu_2.value = mu_2
    if y_1_val is not None:
        prob._y_1_val.value = y_1_val

    prob.solve(verbose=False, solver='ECOS')

    df = pd.DataFrame((prob._adjusted_returns.value, prob._ret_1.value + prob._ret_2.value,
                       prob._risk_1.value + prob._risk_2.value, prob._transaction_cost.value),
                      index=['Risk and Txn Adjusted Return', 'Return', 'Risk', 'Transaction Cost'])

    return df, prob._y_1.value, prob._y_2.value, prob._w_1.value, prob._w_2.value


def evaluate_soln_on_scenarios(y_val, mu1, mu2, single_forecast_problem):
    """
  given a solution evaluate its quality over scenarios
  mu1 and mu2 numpy arrays M x n
  """
    performance = []
    M, num_series = mu1.shape
    for i in range(M):
        df, y_1, y_2, w_1, w_2 = solve_single_forecast_mpc(single_forecast_problem, mu1[i, :], mu2[i, :], y_1_val=y_val)
        performance.append(df.T)
    performance_df = pd.concat(performance, axis=0)

    return performance_df.mean().iloc[0]  # compute the average performance


def solve_multi_forecast_mpc(prob, mu_1, mu_forecasts):
    """
    prob: cvxpy problem intance
    mu1 numpy array vector of expected first stage returns
    mu_forecasts numpy array M by n where M is number of scenarios
    """
    prob._mu1.value = mu_1
    prob._mu_forecasts.value = mu_forecasts

    prob.solve(verbose=False, solver='ECOS')

    df = pd.DataFrame((prob._adjusted_returns.value, prob._expected_returns.value),
                      index=['Stage 1 Objective', 'Stage 2 Objective'])
    return df, prob._y.value, prob._w.value  # today information


def evaluate_portfolio_models(mu1, mu2, cov_matrix, transaction_penalty, risk_aversion):
    """
    Evaluate the difference between multi forecast and deterministic model
    :param mu1:
    :param mu2:
    :param cov_matrix:
    :param transaction_penalty:
    :param risk_aversion:
    :return:
    """
    # evaluate multi forecast
    two_stage_performance = []
    deterministic_performance = []
    M, n = mu2.shape
    predicted_mu1 = mu1.mean(axis=0)
    # generate mu2 using the best scenarios i.e. we do not know which one will occur

    pi = [1 / M for i in range(M)]

    start = time.time()
    # form multi forecast problem
    mf_mpc_prob = define_multi_forecast_mpc(pi, cov_matrix, risk_aversion, transaction_penalty)
    mf_mpc_prob.ignore_dpp = True  # problem is too big and is only solved once
    df_mf, y_val_mf, w_val_mf = solve_multi_forecast_mpc(mf_mpc_prob, predicted_mu1, mu2)
    end = time.time()

    # form single forecast problem
    sf_mpc_prob = define_single_forecast_mpc(cov_matrix, risk_aversion, transaction_penalty, y_1_val_flag=False)

    df_sf, y_val_sf, y_2_sf, w_1_sf, w_2_sf = solve_single_forecast_mpc(sf_mpc_prob, predicted_mu1,
                                                                        mu2.mean(axis=0))

    sf_mpc_prob = define_single_forecast_mpc(cov_matrix, risk_aversion, transaction_penalty, y_1_val_flag=True)

    for scenario in range(M):
        # two stage solution
        df, y_1, y_2, w_1, w_2 = solve_single_forecast_mpc(sf_mpc_prob, mu1[scenario, :], mu2[scenario, :], y_val_mf)
        two_stage_performance.append(df.T)
        # two stage solution
        df, y_1, y_2, w_1, w_2 = solve_single_forecast_mpc(sf_mpc_prob, mu1[scenario, :], mu2[scenario, :], y_val_sf)
        deterministic_performance.append(df.T)

    two_stage_performance_df = pd.concat(two_stage_performance, axis=0)
    deterministic_performance_df = pd.concat(deterministic_performance, axis=0)

    print("Transaction penalty ", transaction_penalty)
    print("VSS ", (two_stage_performance_df.mean().iloc[0] - deterministic_performance_df.mean().iloc[0]))

    print("MFC Time (s) ", end - start)

    return {'deterministic portfolio': w_1_sf, 'deterministic results': deterministic_performance_df,
            'MFC portfolio': w_val_mf, 'MFC results': two_stage_performance_df}


def evaluate_mf(mu1, mu2, prob):
    """
  This function solves the two stage s, risk_aversion, transaction penalty
  and returns the optimal trade
  this function is used to generate the dataset

  mu1 K x N
  mu2 K x N
  cov_matrix: NxN
  transaction penalty = scalar
  risk_aversion = scalar
  """
    # evaluate multi forecast
    M, n = mu2.shape
    predicted_mu1 = mu1.mean(axis=0)
    start = time.time()
    df_mf, y_val, w_val = solve_multi_forecast_mpc(prob, predicted_mu1, mu2)
    end = time.time()
    return df_mf.iloc[0, 0], y_val  # objective value and trade


def evaluation_wrapper(mu1_data_lower, mu2_data_lower, mu1_data_upper,
                       mu2_data_upper, lower_level_problem, upper_level_problem):
    """A wrapper around solve_and_derivative for the batch function."""

    obj_val, y_i = evaluate_mf(mu1_data_lower, mu2_data_lower, lower_level_problem)
    if mu1_data_upper is not None:
        risk_adjusted_return = evaluate_soln_on_scenarios(y_i, mu1_data_upper, mu2_data_upper, upper_level_problem)
    else:
        risk_adjusted_return = obj_val
    return risk_adjusted_return


def batched_program_evaluation(mu1_data_lowers, mu2_data_lowers, mu1_data_uppers,
                               mu2_data_uppers, lower_level_problem, upper_level_problem,
                               n_jobs=-1, pool=None, eval=True):
    """
    Solves a batch of surrogate programs and returns a batched objective value
    Uses a ThreadPool to perform operations across the batch in parallel.

    For more information on the arguments and return values,
    see the docstring for `solve_and_derivative` function.

    Args:

        n_jobs - Number of jobs to use in the forward pass. n_jobs 1
            means serial and n_jobs = -1 defaults to the number of CPUs (default=-1).
        warm_starts - A list of warm starts.
        kwargs - kwargs sent to scs.

    Returns:
        risk_adjusted_returns
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    batch_size = len(mu1_data_lowers)

    n_jobs = min(batch_size, n_jobs)  # which ever is smaller

    if n_jobs == 1:
        # serial
        risk_adjusted_returns = []
        for i in range(batch_size):
            obj_val, y_i = evaluate_mf(mu1_data_lowers[i], mu2_data_lowers[i], lower_level_problem)
            risk_adjusted_return = evaluate_soln_on_scenarios(y_i, mu1_data_uppers[i], mu2_data_uppers[i],
                                                              upper_level_problem)
            risk_adjusted_returns.append(risk_adjusted_return)
    else:
        # thread pool
        # raise NotImplementedError("Multithreading is not implemented yet")
        # pool = ThreadPool(processes=n_jobs)
        # args = [(mu1_data_lower, mu2_data_lower, mu1_data_upper, mu2_data_upper,
        #          lower_level_problem, upper_level_problem) for mu1_data_lower,
        #         mu2_data_lower, mu1_data_upper, mu2_data_upper, lower_level_problem, upper_level_problem in
        #         zip(mu1_data_lowers, mu2_data_lowers, mu1_data_uppers,
        #             mu2_data_uppers, lower_level_problems, upper_level_problems)]
        # with threadpool_limits(limits=1):
        #     results = pool.starmap(evaluation_wrapper, args)
        # pool.close()
        pool_was_none = False
        if pool is None:
            pool_was_none = True
            pool = Pool(processes=n_jobs)
        if eval:
            args = [(mu1_data_lower, mu2_data_lower, mu1_data_upper, mu2_data_upper,
                     lower_level_problem, upper_level_problem) for mu1_data_lower,
                                                                   mu2_data_lower, mu1_data_upper,
                                                                   mu2_data_upper in
                    zip(mu1_data_lowers, mu2_data_lowers, mu1_data_uppers, mu2_data_uppers)]
        else:  # no upper level to evaluate
            args = [(mu1_data_lower, mu2_data_lower, None, None,
                     lower_level_problem, upper_level_problem) for mu1_data_lower,
                                                                   mu2_data_lower in
                    zip(mu1_data_lowers, mu2_data_lowers)]

        results = pool.starmap(evaluation_wrapper, args)

        if pool_was_none:
            pool.close()

        risk_adjusted_returns = [r for r in results]

    return risk_adjusted_returns


def evaluate_predicted_tree(cov_matrix, risk_aversion, txn_penalty, contexts, unique_contexts,
                            predicted_tree, scenarios, n_series, name, cache, experiment_folder,
                            train_time=None, N_sp=None):
    times = []
    if train_time is not None:
        times.append(train_time)
    if N_sp is None:
        N_sp = len(unique_contexts)
    else:
        assert type(N_sp) == int

    mu1_data_lowers = predicted_tree[..., :n_series]
    mu2_data_lowers = predicted_tree[..., n_series:]
    mu1_data_uppers = scenarios.squeeze()[..., :n_series]
    mu2_data_uppers = scenarios.squeeze()[..., n_series:]

    mu1_data_lowers_lst = [mu1_data_lowers[i] for i in range(N_sp)]
    mu2_data_lowers_lst = [mu2_data_lowers[i] for i in range(N_sp)]

    mu1_data_uppers_lst = []
    mu2_data_uppers_lst = []

    for i in range(N_sp):
        indices = np.isclose(contexts, unique_contexts[i]).all(axis=-1).all(axis=-1)
        mu1_data_uppers_lst.append(mu1_data_uppers[indices])
        mu2_data_uppers_lst.append(mu2_data_uppers[indices])

    # evaluate the solutions on the upper level scenarios using our fancy batched method
    mu1 = mu1_data_lowers_lst[0]
    mu2 = mu2_data_lowers_lst[0]
    M, n = mu1.shape  # lower level shape
    pi = [1 / M for i in range(M)]  # equal weight

    mf_mpc_prob = define_multi_forecast_mpc(pi, cov_matrix, risk_aversion, txn_penalty)
    sf_mpc_prob = define_single_forecast_mpc(cov_matrix, risk_aversion, txn_penalty, y_1_val_flag=True)

    n_jobs = mp.cpu_count()
    pool = Pool(n_jobs)

    # solve and evaluation.
    # time solve and evaluation separately
    start = time.time()
    # eval the solutions on the observed distributions
    r = batched_program_evaluation(mu1_data_lowers_lst, mu2_data_lowers_lst, mu1_data_uppers_lst,
                                   mu2_data_uppers_lst, mf_mpc_prob, sf_mpc_prob,
                                   n_jobs=-1, pool=pool)
    end = time.time()

    times.append(end - start)

    start = time.time()
    # just getting the solutions
    batched_program_evaluation(mu1_data_lowers_lst, mu2_data_lowers_lst, mu1_data_uppers_lst,
                               mu2_data_uppers_lst, mf_mpc_prob, sf_mpc_prob,
                               n_jobs=-1, pool=pool, eval=False)
    end = time.time()

    times.append(end - start)
    # evaluation method
    if train_time is None:
        df = pd.DataFrame(times, columns=['time'], index=['eval', 'solve'])
    else:
        df = pd.DataFrame(times, columns=['time'], index=[name + ' mapping training', 'eval', 'solve'])

    df.to_csv(cache + experiment_folder + '/' + name + '_train_solve_eval_times.csv')

    risk_adjusted_returns = pd.Series(r)
    risk_adjusted_returns.to_csv(cache + experiment_folder + '/' + name + '_risk_adjusted_returns.csv')
    pool.close()
    # store returns
    return risk_adjusted_returns, df
