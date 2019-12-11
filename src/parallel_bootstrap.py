import multiprocessing as mp
import numpy as np
import ctypes
from random import randint
import timeit


# def collect_result(result):
#     global results
#     results.append(result)

## Boostrapping without Numpy
def bootstrap(x: list, func, samples: int = 100, sample_size: int = 0) -> list:
    """
    Creates a list of statistics from data points.

    :param x: list with data points from where to
    :param func: function to represent the statistic (ex. mean, std, etc.)
    :param samples: number of samples to take from x
    :param sample_size: the size of each sample, should be between 0 and the total length of x
    """
    if sample_size == 0:
        sample_size = len(x)

    def extract(x):
        rindex = [randint(0, len(x) - 1) for _ in range(sample_size)]
        return [x[i] for i in rindex]

    return [func(extract(x)) for _ in range(samples)]


## Bootstrapping with Numpy
def bootstrap_np(x, func, samples=1000, sample_size=100):
    """


    """
    return [func(np.random.choice(x, sample_size)) for _ in range(samples)]


## Boostrapping in parallel without Numpy
def bootstrap_par(x, func, samples=100, sample_size=0):
    if sample_size == 0:
        sample_size = len(x)

    results = []
    X = mp.Array(ctypes.c_double, len(x))
    X_shr = np.ctypeslib.as_array(X.get_obj())
    X_shr[:] = x

    def bootstrap_complete(sample_size=sample_size):
        rindex = [randint(0, len(x) - 1) for _ in range(sample_size)]
        return func([X_shr[i] for i in rindex])

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(bootstrap_complete,
                           [sample_size for _ in range(samples)])

    pool.close()
    return results


def bootstrap_complete(x, func, sample_size):
    return func(np.random.choice(x, sample_size))


## Bootstrapping in parallel with Numpy
def bootstrap_np_par(x, func, samples=1000, sample_size=100):
    results = []
    X = mp.Array(ctypes.c_double, len(x))
    X_shr = np.ctypeslib.as_array(X.get_obj())
    X_shr[:] = x

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap_async(bootstrap_complete,
                                     [(X_shr, func, sample_size) for _ in range(samples)])

    pool.close()
    return results


if __name__ == '__main__':
    sample_normal = np.random.normal(5, 1, 1000)

    print("Without parallel: \n")
    print(bootstrap(sample_normal, np.mean, 1000, 100)[:5])

    print("\n With parallel: \n")
    print(bootstrap_par(sample_normal, np.mean, samples=1000, sample_size=100))

#     s = '''
# sample_normal = np.random.normal(5, 1, 10000)
# test = bootstrap(sample_normal, np.mean, 1000, 100)
# '''
#
#     setup = '''
# import numpy as np
# from __main__ import bootstrap
# '''
#
#     s_np = '''
# sample_normal = np.random.normal(5, 1, 10000)
# test = bootstrap_np(sample_normal, np.mean, 1000, 100)
# '''
#
#     setup_np = '''
# import numpy as np
# from __main__ import bootstrap_np
# '''
#
#     s_par = '''
# sample_normal = np.random.normal(5, 1, 10000)
# test_p = bootstrap_par(sample_normal, np.mean, 1000, 100)
# '''
#
#     setup_par = '''
# import multiprocessing as mp
# import numpy as np
# from __main__ import bootstrap_par
# '''
#
#     s_np_par = '''
# sample_normal = np.random.normal(5, 1, 10000)
# test_p = bootstrap_np_par(sample_normal, np.mean, 1000, 100)
# '''
#
#     setup_np_par = '''
# import multiprocessing as mp
# import numpy as np
# from __main__ import bootstrap_np_par
# '''
#
#     benchmark = []
#
#     benchmark.append(timeit.Timer(stmt=s, setup=setup).timeit(1))
#     benchmark.append(timeit.Timer(stmt=s_np, setup=setup_np).timeit(1))
#     benchmark.append(timeit.Timer(stmt=s_par, setup=setup_par).timeit(1))
#     benchmark.append(timeit.Timer(stmt=s_np_par, setup=setup_np_par).timeit(1))
#
#
#     print(benchmark)
