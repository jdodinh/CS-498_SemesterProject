from helpers import *
from tqdm import tqdm
import numpy as np
import pickle
import sys




if __name__=="__main__":

    m = int(sys.argv[1])
    j = int(sys.argv[2])
    # print(m)
    # print(j)
    num_vars = 2**m - 1
    num_combos = int(2**num_vars)
    A = make_complete(m, dt=bool).astype(int)[:,1:]
    x = np.zeros(num_vars, dtype=int)
    result_dict = { '2^(m-{}) <= b < 2^(m-{})'.format(k+1, k):0 for k in range(1, m)}
    iter_length = int(num_combos/4)
    for i in tqdm(range(j*iter_length, (j+1)*iter_length)):
        # j = i

        binary = numberToBase(i,2)
        x[num_vars - len(binary):] = binary
        b = A@x
        if (np.all(b[1:]<=b[:-1])):
            for k in range(1, m):
                # print('2^(m-({}+1)) <= b < 2^(m-{})'.format(k+1, k))
                if (np.all(2**(m-(k+1)) <= b) and np.all(b < 2**(m-k))):
                    result_dict['2^(m-{}) <= b < 2^(m-{})'.format(k+1, k)] += 1

    with open('ComputationalResults/ranges'+str(m)+'_'+str(j)+'.pkl', 'wb') as f:
        pickle.dump(result_dict, f)
        f.close()
