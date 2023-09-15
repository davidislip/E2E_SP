#!/usr/bin/env python3

import numpy as np
import gurobipy as gp
import time
from scipy.stats import chi2
global global_flag 
global_flag= 0

def bigm_solver(nr_assets, mu, r_min, matrix_sigma, k, limit_time, MipGap = 0.01):
    
    start = time.time()

    m = gp.Model("miqp")

    m.Params.timeLimit = limit_time
    
    m.Params.MIPGap = MipGap

    b_vars = m.addMVar(nr_assets, vtype=gp.GRB.BINARY, name="b_vars")

    x_vars = m.addMVar(nr_assets, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="x_vars")

    m.setObjective(x_vars @ matrix_sigma @ x_vars, gp.GRB.MINIMIZE)

    m.addConstr(x_vars.sum() == 1)
    
    m.addConstr(mu @ x_vars >= r_min)
    
    m.addConstr(b_vars.sum() <= k)
    
    m.addConstr(x_vars <= b_vars)

    m.optimize()
    
    x = np.zeros(nr_assets)
    for i in range(nr_assets):
        x[i] = x_vars[i].X

    obj_value = m.objVal
    gap1 = m.MIPGap
    gap2 = gap1 * 100
    #best_bound = m.objBound
    end = time.time()
    
    return obj_value, end - start, gap2, x


def bigm_solver_ellipsoid_robust(nr_assets, mu, r_min, matrix_sigma, k, limit_time, MipGap=0.01, cl = 0.99, T = 100):

    e_cf = np.sqrt(chi2.isf(df=nr_assets, q=1 - cl))

    theta = np.diag(np.diag(matrix_sigma))/T

    start = time.time()

    m = gp.Model("miqp")

    m.Params.timeLimit = limit_time

    m.Params.MIPGap = MipGap

    b_vars = m.addMVar(nr_assets, vtype=gp.GRB.BINARY, name="b_vars")

    x_vars = m.addMVar(nr_assets, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="x_vars")

    eta = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name="epi")

    m.setObjective(x_vars @ matrix_sigma @ x_vars + e_cf*eta, gp.GRB.MINIMIZE)

    m.addConstr(x_vars.sum() == 1)

    m.addConstr(mu @ x_vars >= r_min)

    m.addConstr(x_vars @theta @ x_vars <= eta)

    m.addConstr(b_vars.sum() <= k)

    m.addConstr(x_vars <= b_vars)

    m.optimize()

    x = np.zeros(nr_assets)
    for i in range(nr_assets):
        x[i] = x_vars[i].X

    obj_value = m.objVal
    gap1 = m.MIPGap
    gap2 = gap1 * 100
    # best_bound = m.objBound
    end = time.time()

    return obj_value, end - start, gap2, x
