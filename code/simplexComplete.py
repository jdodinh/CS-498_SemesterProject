import cplex
import numpy as np
from helpers import make_complete

def populatebyrow(prob, b):
    m = len(b)
    A = make_complete(m, bool).astype(int)

    my_colnames = np.array(range(2 ** m)).astype(str).tolist()
    my_obj = np.ones(2 ** m).tolist()
    lb = np.zeros(2 ** m).tolist()
    ub = np.ones(2 ** m).tolist()

    prob.variables.add(obj=my_obj, ub=ub, lb=lb, names=my_colnames)

    c_names = ["c" + str(i) for i, b in enumerate(b)]

    constraints = [[my_colnames, A[i, :].tolist()] for i, b in enumerate(b)]

    constraint_senses = ["E" for i in b]

    prob.linear_constraints.add(lin_expr=constraints, senses=constraint_senses, rhs=b.astype(float), names=c_names)

def simplexComplete(b):
    probStat = cplex.Cplex()
    handle = populatebyrow(probStat, b)