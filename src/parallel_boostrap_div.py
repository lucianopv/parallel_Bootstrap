# This file tries to divide the problem
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
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(func, x)
    pool.close()
    return results


if __name__ == "__main__":
    # sample_normal = np.random.normal(5, 1, 1000)
    # result = sampling(sample_normal, 1000, 100)
    # print(bootstrap(result, np.mean)[:5])

    s_100 = '''
sample_normal = np.random.normal(5, 1, 10000)
test = sampling(sample_normal, 100, 100)
test_result = bootstrap(test, np.mean)
'''

    setup = '''
import numpy as np
from __main__ import bootstrap, sampling
'''

    s_par_100 = '''
sample_normal = np.random.normal(5, 1, 10000)
test_p = sampling(sample_normal, 100, 100)
result_p = bootstrap_par(test_p, np.mean)
'''

    setup_par = '''
import multiprocessing as mp
import numpy as np
from __main__ import bootstrap_par, sampling
'''

    s_1000 = '''
sample_normal = np.random.normal(5, 1, 1000)
test = sampling(sample_normal, 1000, 100)
test_result = bootstrap(test, np.mean)
'''

    s_par_1000 = '''
sample_normal = np.random.normal(5, 1, 100000)
test_p = sampling(sample_normal, 1000, 100)
result_p = bootstrap_par(test_p, np.mean)
'''

    s_10000 = '''
sample_normal = np.random.normal(5, 1, 1000)
test = sampling(sample_normal, 100000, 100)
test_result = bootstrap(test, np.mean)
'''

    s_par_10000 = '''
sample_normal = np.random.normal(5, 1, 100000)
test_p = sampling(sample_normal, 100000, 100)
result_p = bootstrap_par(test_p, np.mean)
'''

    benchmark = []

    serial = [timeit.Timer(stmt=i, setup=setup).timeit(10) for i in [s_100, s_1000, s_10000]]
    par = [timeit.Timer(stmt=i, setup=setup_par).timeit(10) for i in [s_par_100, s_par_1000, s_par_10000]]
    repeats = [100, 1000, 100000]

    df = pd.DataFrame(np.c_[serial, par], index=repeats, columns=["Repeats", "Serial", "Parallel"])

    plt.plot(df['Repeats'], df['Serial'], color="red")
    plt.plot(df['Repeats'], df['Parallel'], color="blue")
    plt.show()
