from helpers import *
from tqdm import tqdm
import numpy as np
import pickle
import sys



if __name__=="__main__":
    m = int(sys.argv[1])
    CONDITIONS_ADD = {}
    CONDITIONS_ONE = {}
    for i in range(1, m+1):
        for j in range(1, m+1):
            if i < j:
                CONDITIONS_ADD[f'2^(m-{j}) <= b <= 2^(m-{i})'] = (2**(m-j),2**(m-i))
        CONDITIONS_ONE[f"0 <= b <= 2^(m-{i})"] = 2**(m-i)

    num_proc = int(sys.argv[2])
    j = int(sys.argv[3])
    print(f"Running iteration {str(j)}/{num_proc}")
    feasible = set({})
    max_val = 2 ** (m - 1)
    base = 2 ** (m - 1) + 1
    num_iter = base ** m
    print(f"Overall number of iterations = {num_iter}")
    iter_length = int(num_iter/num_proc)

    b = np.zeros(m, dtype=int)
    A = make_complete(m, int)[:, 1:]

    if j == num_proc:
        ub = max(j * iter_length, num_iter)
    else:
        ub = min(j * iter_length, num_iter)

    print(f'Range: [{(j - 1) * iter_length},{ub})')
    for i in tqdm(range((j-1)*iter_length, ub)):
        if i >= num_iter: break
        b_slice = numberToBase(i, base)
        b[m - len(b_slice):] = b_slice
        prog, x = splx_int(b, A)
        sol = prog.solve()
        if sol is not None:
            feasible.add(b.tolist().__str__())


    with open('ComputationalResults/feasRHS_'+str(m)+'_res'+str(j)+'_'+str(num_proc), 'wb') as f:
        pickle.dump(feasible, f)
        f.close()






