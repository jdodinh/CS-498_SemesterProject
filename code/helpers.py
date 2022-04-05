import numpy as np
import pandas as pd
from time import perf_counter
import cplex
from docplex.mp.model import Model

def make_complete(m, dt):
    if (m==1):
        return np.array([0, 1], dtype=dt)
    else:
        sub_mat = make_complete(m-1, dt)
        zeros = np.zeros([1, np.power(2, m-1)], dtype=dt)
        ones = np.ones([1, np.power(2, m-1)], dtype=dt)
        top = np.hstack([zeros, ones])
        bot = np.hstack([sub_mat, sub_mat])
        return np.vstack([top, bot])

def splx(b):
    m = len(b)
    A = make_complete(m, bool).astype(int)

    my_colnames = np.array(range(2 ** m)).astype(str).tolist()
    my_obj = np.ones(2 ** m).tolist()
    lb = np.zeros(2 ** m).tolist()
    ub = np.ones(2 ** m).tolist()

    prob = cplex.Cplex()

    prob.variables.add(obj=my_obj, ub=ub, lb=lb, names=my_colnames)

    c_names = ["c" + str(i) for i, b in enumerate(b)]

    constraints = [[my_colnames, A[i, :].tolist()] for i, b in enumerate(b)]

    constraint_senses = ["E" for i in b]

    prob.linear_constraints.add(lin_expr=constraints, senses=constraint_senses, rhs=b.astype(float), names=c_names)



    return prob.solve(), prob.solution.get_values()

def splx_rel(b):
    m = len(b)
    model = Model(name="0-1 Matrices Relaxation")
    A = make_complete(m, dt=bool).astype(int)
    x = model.continuous_var_list((2 ** m), name="x", lb=0, ub=1)
    for i in range(m):
        model.add_constraint(sum([x[j] * A[i, j] for j in range(2 ** m)]) == b[i])

    obj_fun = sum(x)
    model.set_objective("min", obj_fun)
    return model, x

def splx_int(b):
    m = len(b)
    model = Model(name="0-1 Matrices Integer")
    A = make_complete(m, dt=bool).astype(int)
    x = model.binary_var_list((2**m), name="x")
    for i in range(m):
        model.add_constraint(sum([x[j]*A[i, j] for j in range(2**m)]) == b[i])

    obj_fun = sum(x)
    model.set_objective("min", obj_fun)
    return model, x

def inc_bin(m):
    for i in range(len(m)-1, -1, -1):
        if m[i] == 0:
            m[i] += 1
            break
        else:
            m[i] = 0
    return m

def find_sol(b):
    m = b.shape[0]
    dt = bool
    A = make_complete(m, dt).astype(int)
    x = np.zeros(2**m, dtype=int)
    sols = []
    for i in range(2**2**m):
        if np.array_equal(A@x, b):
            sols.append((np.sum(x), x.copy()))
        x = inc(x)
    return sols
