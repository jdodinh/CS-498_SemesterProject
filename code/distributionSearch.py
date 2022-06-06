from helpers import *
from tqdm import tqdm
import numpy as np
import pickle
import sys
# import collections
# import matplotlib.pyplot as plt



if __name__=="__main__":
    m = int(sys.argv[1])

    results = {}
    num_proc = int(sys.argv[2])
    j = int(sys.argv[3])
    print(f"Running iteration {str(j)}/{num_proc}")

    num_vars = 2 ** m - 1

    num_combos = int(2 ** num_vars)
    A = make_complete(m, dt=bool).astype(int)[:, 1:]
    x = np.zeros(num_vars, dtype=int)
    iter_length = int(num_combos / num_proc)
    print(f'Range: [{((j - 1) * iter_length)},{((j) * iter_length)})')
    for i in tqdm(range((j - 1) * iter_length, (j) * iter_length)):
        binary = numberToBase(i, 2)
        x[num_vars - len(binary):] = binary
        b = A @ x
        sum = b.sum()
        if sum in results.keys():
            results[sum] += 1
        else:
            results[sum] = 1

    # for item in sorted(results.items()):
    #     print(item)
    # plt.bar(results.keys(), results.values())
    # plt.show()

    with open(f'ComputationalResults/distrSearch/distr{m}_{j}_{num_proc}.pkl', 'wb') as f:
        pickle.dump(results, f)
        f.close()


