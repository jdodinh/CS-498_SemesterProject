import numpy as np
from tqdm import tqdm
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

def splx_rel(A, b):
    m = len(b)
    model = Model(name="0-1 Matrices Relaxation")
    # A = make_complete(m, dt=bool).astype(int)
    x = model.continuous_var_list((2 ** m), name="x", lb=0, ub=1)
    for i in range(m):
        model.add_constraint(sum([x[j] * A[i, j] for j in range(2 ** m)]) == b[i])

    obj_fun = sum(x)
    model.set_objective("min", obj_fun)
    return model, x

def splx_int(b, A=None):
    if A is None:
        A = make_complete(len(b), int)[:, 1:]
    m = len(b)
    model = Model(name="0-1 Matrices Integer")
    # A = make_complete(m, dt=bool).astype(int)
    x = model.binary_var_list((2**m-1), name="x")
    for i in range(m):
        model.add_constraint(sum([x[j]*A[i, j] for j in range(2**m-1)]) == b[i])

    obj_fun = sum(x)
    model.set_objective("min", obj_fun)
    return model, x

def splx_int_trunc(b, A=None):
    if A is None:
        A = make_complete(len(b), int)[:, 1:-1]
    m = len(b)
    model = Model(name="0-1 Matrices Integer")
    # A = make_complete(m, dt=bool).astype(int)
    x = model.binary_var_list((2**m-2), name="x")
    for i in range(m):
        model.add_constraint(sum([x[j]*A[i, j] for j in range(2**m-2)]) == b[i])

    obj_fun = sum(x)
    model.set_objective("min", obj_fun)
    return model, x

def spx_model(name):
    return Model(name=name)


def splx_int_mult(A, b, name="splx_int_mult"):
    m = len(b)
    model = Model(name=name)
    x = model.binary_var_list((2**m), name="x")
    for i in range(m):
        model.add_constraint(sum([x[j]*A[i, j] for j in range(2**m)]) == b[i])

    obj_fun = sum(x)
    model.set_objective("min", obj_fun)
    model.parameters.mip.pool.intensity.set(4)
    model.parameters.mip.pool.absgap.set(0)
    model.parameters.mip.pool.relgap.set(0)
    model.parameters.mip.limits.populate.set(2)
    sol_pool = model.populate_solution_pool()
    return model, sol_pool.size

def splx_IP_mult(A, b, name="splc_IP_mult"):
    m = len(b)
    model = Model(name=name)
    x = model.binary_var_list(((2**m-1)), name="x")
    for i in range(m):
        model.add_constraint(sum([x[j]*A[i, j] for j in range((2**m-1))]) == b[i])

    # obj_fun = sum(x)
    # model.set_objective("min", obj_fun)

    model.parameters.mip.pool.intensity.set(4)
    model.parameters.mip.pool.absgap.set(0)
    model.parameters.mip.pool.relgap.set(0)
    model.parameters.mip.limits.populate.set((2**(2*(m-1))))
    model.parameters.mip.pool.capacity.set(2**(2**(m-1)))
    return model, x

def inc_bin(m):
    for i in range(len(m)-1, -1, -1):
        if m[i] == 0:
            m[i] += 1
            break
        else:
            m[i] = 0
    return m

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return np.array(digits[::-1], dtype=int)


def countSortedArrays(n, m):
    # Create an array of size M+1
    dp = [0 for _ in range(m + 1)]

    # Base cases
    dp[0] = 1

    # Fill the dp table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # dp[j] will be equal to maximum
            # number of sorted array of size j
            # when elements are taken from 1 to i
            dp[j] = dp[j - 1] + dp[j]

        # Here dp[m] will be equal to the
        # maximum number of sorted arrays when
        # element are taken from 1 to i

    # Return the result
    return dp[m]


# Driver code
# Given Input
# n = 2
# m = 3
#
# # Function Call
# print(countSortedArrays(n, m))

def enumerateSorted(max, len):
    res = []
    for i in range(max+1):
        if len > 1:
            sub_prob = enumerateSorted(i, len-1)
            for sb in sub_prob:
                res.append([i]+sb)
        else:
            res.append([i])
    return res

def find_min_feas(m):
    min_feas = []
    A = make_complete(m, bool).astype(int)
    A_sh = A[:,1:-1]
    b_counter = np.zeros(2**m-2, dtype=int)
    b_ones = np.ones(m, dtype=int)
    for i in tqdm(range(2**(2**m-2))):
        binary = np.array(list(bin(i).replace("0b", "")), dtype=int)
        b_counter[2**m-2-len(binary):] = binary
        b = A_sh@b_counter
        int_prog, vars = splx_int(A, b-b_ones)
        if (int_prog.solve() is None):
            min_feas.append(b)

    return min_feas

def find_min_feas2(m):
    min_feas = []
    b_ones = np.ones(m, dtype=int)
    A = make_complete(m, bool).astype(int)
    for b in tqdm(enumerateSorted(2**(m-1), m)):
        # if (2**(m-1) in b): continue
        int_prog1, vars1 = splx_int(A, b)
        int_prog2, vars2 = splx_int(A, b - b_ones)
        if (int_prog1.solve() is not None and int_prog2.solve() is None):
            min_feas.append(b)
    return min_feas

def enumerate_frac(m):

    A = make_complete(m, bool).astype(int)[:, 1:]
    b_counter = np.zeros(2**m-1, dtype=int)
    for i in tqdm(range(2**(2**m-1))):
        binary = np.array(list(bin(i).replace("0b", "")), dtype=int)
        b_counter[2**m-1-len(binary):] = binary
        b = A@b_counter
        int_prog, vars = splx_int(b)
        rel_prog, vars = splx_rel(b)
        if ((int_prog.solve() is not None) and (rel_prog.solve() is None)):
            print("Found counter example: ", b)
        # Solve for the solutions of Ab, finding all the solutions, and also the support minimal ones.
        # Add the ones that have two support minimal solutions.



