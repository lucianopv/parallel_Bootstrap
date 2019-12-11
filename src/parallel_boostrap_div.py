## This file tries to divide the problem
import numpy as np
import multiprocessing as mp
import timeit
import pandas as pd
import matplotlib.pyplot as plt


def sampling(x, samples=1000, sample_size=100):
    return np.array([np.random.choice(x, sample_size) for _ in range(samples)])


def bootstrap(x, func):
    return np.array([func(z) for z in x])


def bootstrap_par(x, func):
    with mp.Pool(processes=2) as pool:
        results = pool.map(func, x, chunksize=len(x) // 2 or 1)
    pool.close()
    return results


if __name__ == "__main__":
    # sample_normal = np.random.normal(5, 1, 1000)
    # result = sampling(sample_normal, 1000, 100)
    # print(bootstrap(result, np.mean))

    s = '''
sample_normal = np.random.normal(5, 1, 10000)
test = sampling(sample_normal, 1000, 100)
test_result = bootstrap(test, np.mean)
'''

    setup = '''
import numpy as np
from __main__ import bootstrap, sampling
'''

    s_par = '''
sample_normal = np.random.normal(5, 1, 10000)
test_p = sampling(sample_normal, 1000, 100)
result_p = bootstrap_par(test_p, np.mean)
'''

    setup_par = '''
import multiprocessing as mp
import numpy as np
from __main__ import bootstrap_par, sampling
'''

    benchmark = []

    serial = [timeit.Timer(stmt=s, setup=setup).timeit(i) for i in range(1, 100, 10)]
    par = [timeit.Timer(stmt=s_par, setup=setup_par).timeit(i) for i in range(1, 100, 10)]
    repeats = range(1, 1000, 100)

    data = {"Repeats": repeats, "Serial": serial, "Parallel": par}

    df = pd.DataFrame(data)

    plt.plot(df['Repeats'], df['Serial'])
    plt.plot(df['Repeats'], df['Parallel'])
    plt.show()
