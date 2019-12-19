# This file tries to divide the problem
import numpy as np
import multiprocessing as mp
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

    s = '''
sample_normal = np.random.normal(5, 1, 10000)
test = sampling(sample_normal, {}, 100)
test_result = bootstrap(test, np.mean)
'''

    setup = '''
import numpy as np
from __main__ import bootstrap, sampling
'''

    s_par = '''
sample_normal = np.random.normal(5, 1, 10000)
test_p = sampling(sample_normal, {}, 100)
result_p = bootstrap_par(test_p, np.mean)
'''

    setup_par = '''
import multiprocessing as mp
import numpy as np
from __main__ import bootstrap_par, sampling
'''

    instructions_s = [s.format(i) for i in range(1000, 1000000, 10000)]
    instructions_s_par = [s_par.format(i) for i in range(1000, 1000000, 10000)]
    serial = [timeit.Timer(stmt=ins, setup=setup).timeit(1) for ins in instructions_s]
    par = [timeit.Timer(stmt=ins, setup=setup_par).timeit(1) for ins in instructions_s_par]
    repeats = range(1000, 1000000, 10000)

    df = pd.DataFrame(np.c_[serial, par], index=repeats, columns=['Serial', 'Parallel'])

    df.to_pickle('./data/DivPar.pkl')

    sns.set(style='darkgrid', palette='Paired')
    sns.lineplot(data=df, dashes=[(None, None), (2, 2)])
    plt.legend(labels=['Serial', 'Parallel'])
    plt.show()
