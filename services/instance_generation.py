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

    jp = (0.5)*(jp/jp.sum(axis = 0)).T + (0.5)*(jp/jp.sum(axis = 0))
    return jp


def generate_mu1(x, beta0 = 1,  beta1 = 1, beta2 = 0.01, beta3 = 0.01):
    return beta0*x + beta1*x**2 - beta2*((x < -0.0).astype(float)) + beta3*((x > 0.12/12).astype(float))

def generate_mu2_scenarios(x, mu1, y, impact = 0.5):
    return (1 + x)*mu1 + impact*x*y