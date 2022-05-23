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
                CONDITIONS_ADD[f'2^(m-{j}) <= b <= 2^(m-{i}), UB'] = (2**(m-j),2**(m-i), 'UB')
                CONDITIONS_ADD[f'2^(m-{j}) <= b <= 2^(m-{i}), LB'] = (2 ** (m - j), 2 ** (m - i), 'LB')
        CONDITIONS_ONE[f"0 <= b <= 2^(m-{i}), UB"] = (2**(m-i), 'UB')


    num_proc = int(sys.argv[2])
    j = int(sys.argv[3])
    print(f"Running iteration {str(j)}/{num_proc}")
    # print(m)
    # print(j)
    num_vars = 2**m - 1
    num_combos = int(2**num_vars)
    A = make_complete(m, dt=bool).astype(int)[:,1:]
    x = np.zeros(num_vars, dtype=int)
    result_dict = { k:0 for k in CONDITIONS_ADD.keys()}
    for k in CONDITIONS_ONE.keys(): result_dict[k] = 0
    iter_length = int(num_combos/num_proc)
    print(f'Range: [{((j-1)*iter_length)},{((j)*iter_length)})')
    for i in tqdm(range((j-1)*iter_length, (j)*iter_length)):
        # j = i

        binary = numberToBase(i,2)
        x[num_vars - len(binary):] = binary
        b = A@x
        # if (np.all(b[1:]<=b[:-1])):
        for (k, v) in CONDITIONS_ADD.items():
            # print('2^(m-({}+1)) <= b < 2^(m-{})'.format(k+1, k))
            if (np.all(v[0] <= b) and np.all(b <= v[1])):
                if v[2] == 'UB' and np.any(b == v[1]):
                    # if np.any(b==v[1]):
                    #     if (np.all(v[0] <= b) and np.all(b <= v[1]) and np.any(b==v[1])):
                    result_dict[k] += 1
                if v[2] == 'LB' and np.any(b == v[0]):
                    # if np.any(b==v[0]):
                    #     if (np.all(v[0] <= b) and np.all(b <= v[1])):
                    result_dict[k] += 1
        for (k, v) in CONDITIONS_ONE.items():
            # print('2^(m-({}+1)) <= b < 2^(m-{})'.format(k+1, k))
            if np.all(b <= v[0]) and np.any(b == v[0]):
                result_dict[k] += 1


    with open('ComputationalResults/ranges'+str(m)+'_'+str(j)+'.pkl', 'wb') as f:
        pickle.dump(result_dict, f)
        f.close()
