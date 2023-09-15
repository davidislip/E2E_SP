#!/usr/bin/env python3

import gurobipy as gp
from numpy import linalg as LA
import numpy as np
import time

global global_flag 
global_flag= 0

def create_l1_problem(nr_assets, mu, r_min):
    # Create an empty model
    m = gp.Model("l1_portfolio")
    m.params.OutputFlag = global_flag
    x_vars = m.addMVar(nr_assets, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="x_vars")

    y_plus_vars = m.addMVar(nr_assets, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="y_plus_vars")

    y_minus_vars = m.addMVar(nr_assets, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="y_minus_vars")

    # Set constraints
    m.addConstr(x_vars.sum() == 1)
    m.addConstr(mu @ x_vars >= r_min)
    m.update()
    return m, x_vars, y_plus_vars, y_minus_vars


def solve_l1_problem(m, x_vars, y_plus_vars, y_minus_vars, nr_assets, matrix_sigma, penalty_param, w):
    c = m.addConstr(x_vars - w == y_plus_vars - y_minus_vars)
    
    # Set objective
    vec_penalty_param = np.full(nr_assets, penalty_param)
    m.setObjective(x_vars @ matrix_sigma @ x_vars + vec_penalty_param @ y_plus_vars
                   + vec_penalty_param @ y_minus_vars, gp.GRB.MINIMIZE)
    ##m.write("out.lp")
    m.optimize()

    # Extract solution
    x = np.zeros(nr_assets)
    for i in range(nr_assets):
        x[i] = x_vars[i].X

    m.remove(c)
    m.update()

    return x


def solve_discrete_problem(w_zeros, x, k):
    k_largest = np.argsort(x)[len(x) - k:]
    for key in k_largest:
        w_zeros[key] = x[key]
    w_zeros = w_zeros / sum(w_zeros)

    return w_zeros


def partial_minimum(x_new, x, w_new, w, tol):
    dif1 = x_new - x
    dif2 = w_new - w
    dif = np.concatenate((dif1, dif2))
    return LA.norm(dif, np.inf) < tol


def coupling_satisfied(x_new, w_new, tol):
    dif = x_new - w_new
    return LA.norm(dif, 1) <= tol


def padm(nr_assets, vec_returns, expected_ret, matrix_sigma, k, penalty_param, time_limit, total_iteration_limit = 20, check_coupling = False):
    start = time.time()

    m, x_vars, y_plus_vars, y_minus_vars = create_l1_problem(nr_assets, vec_returns, expected_ret)

    w_zeros = np.zeros(nr_assets)
    w = w_zeros

    # Initialize outer iteration count
    total_iter_counter = 1
    adm_iterations = []
    penalty_params = []
    while penalty_param < 1e10 and total_iter_counter <= total_iteration_limit:
        # Initialize inner iteration counter
        inner_iter_counter = 1
      
        while inner_iter_counter < 1e8 and total_iter_counter <= total_iteration_limit:
            x_new = solve_l1_problem(m, x_vars, y_plus_vars, y_minus_vars, nr_assets, matrix_sigma, penalty_param, w)

            w_new = solve_discrete_problem(w_zeros, x_new, k)

            # Check for partial minimum
            if inner_iter_counter > 1 and partial_minimum(x_new, x, w_new, w, tol=1e-5):
                break
            w = w_new
            x = x_new
            inner_iter_counter += 1
            total_iter_counter += 1
            penalty_params.append(penalty_param)
            
        adm_iterations.append(inner_iter_counter)
        
        
        if (check_coupling and 
        (coupling_satisfied(x_new, w_new, tol=1e-5) or time.time() - start >= time_limit)):
            break
        #if the coupling is not satisfied we must increment 
        if not coupling_satisfied(x_new, w_new, tol=1e-5):
            penalty_param *= 10

    obj_value = x_new @ matrix_sigma @ x_new

    end = time.time()
    return adm_iterations, total_iter_counter, end - start, obj_value, penalty_params, x_new
