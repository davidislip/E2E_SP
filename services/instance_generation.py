import numpy as np


def vineBeta(d, betaparam):
    P = np.zeros((d, d))  # storing partial correlations
    S = np.eye(d)

    for k in range(1, d):
        for i in range(k + 1, d):
            P[k, i] = np.random.beta(betaparam, betaparam)  # sampling from beta
            P[k, i] = (P[k, i] - 0.5) * 2  # linearly shifting to [-1, 1]
            p = P[k, i]
            for l in range(k - 1, -1, -1):  # converting partial correlation to raw correlation
                p = p * np.sqrt((1 - P[l, i] ** 2) * (1 - P[l, k] ** 2)) + P[l, i] * P[l, k]
            S[k, i] = p
            S[i, k] = p

    # permuting the variables to make the distribution permutation-invariant
    permutation = np.random.permutation(d)
    S = S[permutation][:, permutation]

    return S


def cholesky_decomposition(matrix):
    L = np.linalg.cholesky(matrix)
    return L


def generate_uniform_random_vector(lower_bound, upper_bound, size):
    vector = np.random.uniform(lower_bound, upper_bound, size)
    return vector


def correlation_to_covariance(std_devs, correlation):
    covariance = np.outer(std_devs, std_devs) * correlation
    return covariance


def sample_from_multivariate_normal(mean, covariance, num_samples):
    samples = np.random.multivariate_normal(mean, covariance, num_samples)
    return samples


def generate_joint_probabilities(diagonal_values):
    n = len(diagonal_values)
    off_diagonal_values = np.random.uniform(0, 1, size=(n, n))
    off_diagonal_values = (off_diagonal_values + off_diagonal_values.T) / 2  # Make it symmetric

    matrix = np.diag(diagonal_values) + off_diagonal_values
    row_sums = np.sum(matrix, axis=1) - np.diag(matrix)
    correction_matrix = np.diag(row_sums)
    jp = matrix - np.diag(np.diag(matrix)) + correction_matrix

    jp = (0.5) * (jp / jp.sum(axis=0)).T + (0.5) * (jp / jp.sum(axis=0))
    return jp


def generate_mu1(x, beta0=1, beta1=1, beta2=0.01, beta3=0.01):
    return beta0 * x + beta1 * x ** 2 - beta2 * ((x < -0.0).astype(float)) + beta3 * ((x > 0.12 / 12).astype(float))


def generate_mu2_scenarios(x, mu1, y, impact=0.5):
    return (1 + x) * mu1 + impact * x * y


def simulate_var_process(num_obs, num_series, mean_return, cov_matrix, ar_order, replications, ar_coeffs=None):
    if len(mean_return) != num_series or cov_matrix.shape != (num_series, num_series):
        raise ValueError("Invalid dimensions for mean_return or cov_matrix")

    num_lags = ar_order
    num_total_obs = num_obs + num_lags

    # Generate random innovations from a multivariate normal distribution
    innovations = np.random.multivariate_normal(mean=mean_return, cov=cov_matrix, size=num_total_obs)

    # Initialize data matrix to store the generated observations
    data = np.zeros((replications, num_total_obs, num_series))

    # Fill in the initial observations with random innovations
    for replication in range(replications):
        data[replication, :num_lags, :] = innovations[:num_lags, :]  # same data for every starting sequence

    if ar_coeffs is None:
        ar_coeffs = -0.8 + 0.2 * np.random.random(num_series * num_lags).reshape((num_lags, num_series))

    for replication in range(replications):
        innovations = np.random.multivariate_normal(mean=mean_return, cov=cov_matrix, size=num_total_obs)
        # Generate the VAR process
        for t in range(num_lags, num_total_obs):
            lagged_data = data[replication, t - num_lags:t, :]
            data[replication, t, :] = (lagged_data * ar_coeffs).sum(axis=0) + innovations[t, :]

    # Discard the initial observations used for lagged values
    # data = data[num_lags:, :]

    return data, ar_coeffs


def retrive_training_data(cache, experiment_folder, sampling_nums):
    targets = []
    mu1_data_lowers = []
    mu2_data_lowers = []
    inputs = []
    mu1_data_uppers = []
    mu2_data_uppers = []
    contexts = []
    for sampling_num in sampling_nums:
        sampling_dir = cache + experiment_folder + str(sampling_num) + "/"
        mu1_data_lower = np.load(sampling_dir + 'surrogate_scenarios_mu1.npy')
        mu2_data_lower = np.load(sampling_dir + 'surrogate_scenarios_mu2.npy')
        mu1_data_upper = np.load(sampling_dir + 'evaluation_scenarios_mu1.npy')
        mu2_data_upper = np.load(sampling_dir + 'evaluation_scenarios_mu2.npy')
        context = np.load(sampling_dir + 'contexts.npy')
        target = np.load(sampling_dir + 'optimal_objectives.npy')

        targets.append(target)
        # get scenario data for the lower level problem
        lower_level_inputs = np.concatenate([mu1_data_lower, mu2_data_lower], axis=-1)
        # get scenario data for the upper level problem
        upper_level_inputs = np.concatenate([mu1_data_upper, mu2_data_upper], axis=-1)
        input_ = np.concatenate([lower_level_inputs, upper_level_inputs],
                                axis=1)  # the upper level scenario is the last spot

        inputs.append(input_)
        mu1_data_lowers.append(mu1_data_lower)
        mu2_data_lowers.append(mu2_data_lower)
        mu1_data_uppers.append(mu1_data_upper)
        mu2_data_uppers.append(mu2_data_upper)
        contexts.append(context)

    targets = np.concatenate(targets, axis=0)
    mu1_data_lowers = np.concatenate(mu1_data_lowers, axis=0)
    mu2_data_lowers = np.concatenate(mu2_data_lowers, axis=0)
    mu1_data_uppers = np.concatenate(mu1_data_uppers, axis=0)
    mu2_data_uppers = np.concatenate(mu2_data_uppers, axis=0)
    contexts = np.concatenate(contexts, axis=0)
    inputs = np.concatenate(inputs, axis=0)

    return inputs, targets, mu1_data_lowers, mu2_data_lowers, mu1_data_uppers, mu2_data_uppers, contexts
