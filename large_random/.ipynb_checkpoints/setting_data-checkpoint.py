#!/usr/bin/env python3

import numpy as np
import gurobipy as gp

global global_flag 
global_flag= 0

def compute_minimum_portfolio(nr_assets, matrix_sigma):
    # Create an empty model
    
    m = gp.Model("min_risk")
    m.params.OutputFlag = global_flag

    # Add a variable for each asset
    x_vars = m.addMVar(nr_assets, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x_vars")

    # Set objective
    m.setObjective(x_vars @ matrix_sigma @ x_vars, gp.GRB.MINIMIZE)

    # Set constraints
    m.addConstr(x_vars.sum() == 1)

    m.optimize()

    # Extract solution
    x_min = np.zeros(nr_assets)
    for i in range(nr_assets):
        x_min[i] = x_vars[i].X

    return x_min


def compute_mean_variance_portfolio(nr_assets, matrix_sigma, r_min, mean_vec_ret):
    # Create an empty model
    m = gp.Model("mvo  ")
    m.params.OutputFlag = global_flag
    # Add a variable for each asset
    x_vars = m.addMVar(nr_assets, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x_vars")

    # Set objective
    m.setObjective(x_vars @ matrix_sigma @ x_vars, gp.GRB.MINIMIZE)

    # Set constraints
    m.addConstr(x_vars.sum() == 1)
    m.addConstr(mean_vec_ret @ x_vars >= r_min)
    
    m.optimize()

    # Extract solution
    x_min = np.zeros(nr_assets)
    for i in range(nr_assets):
        x_min[i] = x_vars[i].X

    return x_min


def compute_maximum_portfolio(nr_assets, mean_vec_ret):
    m = gp.Model("max_return")
    m.params.OutputFlag = global_flag
    # Add a variable for each asset
    x_vars = m.addMVar(nr_assets, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="x_vars")

    m.setObjective(mean_vec_ret @ x_vars, gp.GRB.MAXIMIZE)

    m.addConstr(x_vars.sum() == 1)

    m.optimize()

    # Extract solution
    x_max = np.zeros(nr_assets)
    for i in range(nr_assets):
        x_max[i] = x_vars[i].X

    return x_max


def set_minimum_return(nr_assets, mean_ret_vec, matrix_sigma):
    x_min = compute_minimum_portfolio(nr_assets, matrix_sigma)
    x_max = compute_maximum_portfolio(nr_assets, mean_ret_vec)
    r_min = mean_ret_vec @ x_min
    r_max = mean_ret_vec @ x_max
    exp_ret = r_min + 0.3 * (r_max - r_min)

    return exp_ret


def get_data(dir_, name):
    with open(dir_ + 'mu/' + name, "r") as data_file:
        data_lines_ret = data_file.readlines()
    mean_return_entry = []
    for data_point in data_lines_ret:
        entry = data_point.split()
        scalar = float(entry[0])
        np_scalar = np.array(scalar)
        mean_return_entry.append(np_scalar)
    mean_returns_vector = np.array(mean_return_entry)

    nr_of_assets = int(len(mean_returns_vector))

    rows_sigma_list = []
    with open(dir_ + 'Q/' + name, "r") as data_file_sigma:
        lines_sigma = data_file_sigma.readlines()
    for data_point in lines_sigma:
        splitted_line = data_point.split()
        vector = []
        for entry in splitted_line:
            vector.append(float(entry))
        np_vector = np.array(vector)
        rows_sigma_list.append(np_vector)

    covariance_matrix = np.array(rows_sigma_list)

    expected_ret = set_minimum_return(nr_of_assets, mean_returns_vector, covariance_matrix)

    return nr_of_assets, mean_returns_vector, covariance_matrix, expected_ret


