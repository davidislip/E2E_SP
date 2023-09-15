#!/usr/bin/env python3

import numpy as np
import gurobipy as gp
import time


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
