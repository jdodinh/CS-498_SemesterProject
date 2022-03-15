import numpy as np
import pandas as pd
from time import perf_counter

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

if __name__=="__main__":
    m = 28
    dt = bool

    time_start = perf_counter()
    complete_mat = make_complete(m, dt=dt)
    time_stop = perf_counter()

    print("m =", m, "\t Datatype =", dt)
    print("Time taken =", 1000*(time_stop-time_start), "ms")
