import multiprocessing as mp
from multiprocessing.pool import ApplyResult

import numpy as np
import ctypes
from random import randint
import timeit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Callable, List

# def collect_result(result):
#     global results
#     results.append(result)

# Boostrapping without Numpy
from traitlets import List


def bootstrap(x: np.ndarray, func: Callable, samples: int = 100, sample_size: int = 0) -> list:
    """
    This function generates a determined number of samples from an initial array and apply to each the
    function of the statistic. It uses the random library to randomize the indexes for sampling.

    Parameters
    ----------
    x : np.ndarray
        Structure of data to apply the bootstrapping process
    func : function
        Function of how to apply the statistic (ex. mean, std, etc.)
    samples : int
        Number of samples to take from x
    sample_size :   int
        Size of each sample, should be between 0 and the total length of x

    Returns
    -------
    list
        A list with the result of applying the function to each sample

    Examples
    --------
    Generate a bootstrap distribution of the mean of a sample from a normal distribution(5, 1)
    >>> sample_normal = np.random.normal(5, 1, 1000)
    >>> bootstrap(sample_normal, np.mean, samples = 1000, sample_size = 100)
    """

    assert isinstance(x, np.ndarray), "Please convert the input into an numpy ndarray"

    if sample_size == 0:
        sample_size = len(x)

    def extract(y):
        rindex = [randint(0, len(x) - 1) for _ in range(sample_size)]
        return [y[i] for i in rindex]

    return [func(extract(x)) for _ in range(samples)]


# Bootstrapping with Numpy
def bootstrap_np(x: np.ndarray, func: Callable, samples: int = 1000, sample_size: int = 100) -> list:
    """
    This function generates a determined number of samples from an initial array and apply to each the
    function of the statistic. It uses Numpy to reduce the running time.

    Parameters
    ----------
    x : np.ndarray
        Structure of data to apply the bootstrapping process
    func : function
        Function of how to apply the statistic (ex. mean, std, etc.)
    samples : int
        Number of samples to take from x
    sample_size :   int
        Size of each sample, should be between 0 and the total length of x

    Returns
    -------
    list
        A list with the result of applying the function to each sample

    Examples
    --------
    Generate a bootstrap distribution of the mean of a sample from a normal distribution(5, 1)
    >>> sample_normal = np.random.normal(5, 1, 1000)
    >>> bootstrap_np(sample_normal, np.mean, samples = 1000, sample_size = 100)
    """
    return [func(np.random.choice(x, sample_size)) for _ in range(samples)]


def bootstrap_par_comp(x: np.ndarray, func: Callable, sample_size: int) -> float:
    """
    This function generates a determined number of samples from an initial array and apply to each the
    function of the statistic. It does not uses numpy to randomize the samples. This function will be used
    to define a new function which uses parallel computing. It is define outside the parallel function to
    avoid error from the multiprocessing module.

    Parameters
    ----------
    x : np.ndarray
        Structure of data to apply the bootstrapping process
    func : function
        Function of how to apply the statistic (ex. mean, std, etc.)
    sample_size :   int
        Size of each sample, should be between 0 and the total length of x

    Returns
    -------
    float
        A float as a result of the function applied to the specific samples size.

    Examples
    --------
    Generate a sample of a normal distribution and estimate the mean of it.
    >>> sample_normal = np.random.normal(5, 1, 1000)
    >>> bootstrap_par_comp(sample_normal, np.mean, sample_size = 100)
    """
    rindex = [randint(0, len(x) - 1) for _ in range(sample_size)]
    return func([x[i] for i in rindex])


# Boostrapping in parallel without Numpy
def bootstrap_par(x: np.ndarray, func: Callable, samples: int = 100, sample_size: int = 0) -> list:
    """
    This function generates a determined number of samples from an initial array and apply to each the
    function of the statistic. It uses the multiprocessing module to create a pool of workers to divide the
    input into equal sizes (i.e. divides the list of samples). Important to notice that the structure of data
    is created as a memory shared array to avoid incurring in creating multiple copies of it.

    Parameters
    ----------
    x : np.ndarray
        Structure of data to apply the bootstrapping process
    func : function
        Function of how to apply the statistic (ex. mean, std, etc.)
    samples : int
        Number of samples to take from x
    sample_size :   int
        Size of each sample, should be between 0 and the total length of x

    Returns
    -------
    list
        A list with the result of applying the function to each sample

    Examples
    --------
    Generate a bootstrap distribution of the mean of a sample from a normal distribution(5, 1)
    >>> sample_normal = np.random.normal(5, 1, 1000)
    >>> bootstrap_par(sample_normal, np.mean, samples = 1000, sample_size = 100)
    """
    if sample_size == 0:
        sample_size = len(x)

    x_ = mp.Array(ctypes.c_double, len(x))
    x_shr = np.ctypeslib.as_array(x_.get_obj())
    x_shr[:] = x

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap_async(bootstrap_par_comp,
                                     [(x_shr, func, sample_size) for _ in range(samples)]).get()

    pool.close()
    return results


def bootstrap_complete(x: np.ndarray, func: Callable, sample_size: int) -> float:
    """
    This function generates a determined number of samples using the module random of numpy from an initial array
    and apply to each the function of the statistic. The function will be used to define a new function which
    uses parallel computing. It is define outside the parallel function to avoid error from the multiprocessing
    module.

    Parameters
    ----------
    x : np.ndarray
        Structure of data to apply the bootstrapping process
    func : function
        Function of how to apply the statistic (ex. mean, std, etc.)
    sample_size :   int
        Size of each sample, should be between 0 and the total length of x

    Returns
    -------
    float
        A float as a result of the function applied to the specific samples size.

    Examples
    --------
    Generate a sample of a normal distribution and estimate the mean of it.
    >>> sample_normal = np.random.normal(5, 1, 1000)
    >>> bootstrap_complete(sample_normal, np.mean, sample_size = 100)
    """
    return func(np.random.choice(x, sample_size))


# Bootstrapping in parallel with Numpy
def bootstrap_np_par(x: np.ndarray, func: Callable, samples: int = 1000, sample_size: int = 100):
    """
     This function generates a determined number of samples from an initial array and apply to each the
     function of the statistic. It uses the multiprocessing module to create a pool of workers to divide the
     input into equal sizes (i.e. divides the list of samples). Important to notice that the structure of data
     is created as a memory shared array to avoid incurring in creating multiple copies of it.

     Parameters
     ----------
     x : np.ndarray
         Structure of data to apply the bootstrapping process
     func : function
         Function of how to apply the statistic (ex. mean, std, etc.)
     samples : int
         Number of samples to take from x
     sample_size :   int
         Size of each sample, should be between 0 and the total length of x

     Returns
     -------
     list
         A list with the result of applying the function to each sample

     Examples
     --------
     Generate a bootstrap distribution of the mean of a sample from a normal distribution(5, 1)
     >>> sample_normal = np.random.normal(5, 1, 1000)
     >>> bootstrap_par(sample_normal, np.mean, samples = 1000, sample_size = 100)
     """
    x_ = mp.Array(ctypes.c_double, len(x))
    x_shr = np.ctypeslib.as_array(x_.get_obj())
    x_shr[:] = x

    with mp.Pool(mp.cpu_count()) as pool:
        results: ApplyResult[List[float]] = pool.starmap_async(bootstrap_complete,
                                                               [(x_shr, func, sample_size) for _ in range(samples)])

    pool.close()
    return results


if __name__ == '__main__':
    # sample_normal = np.random.normal(5, 1, 1000)
    #
    # print("Without parallel: \n")
    # print(bootstrap(sample_normal, np.mean, 1000, 100)[:5])
    #
    # print("\n With parallel: \n")
    # print(bootstrap_par(sample_normal, np.mean, samples=1000, sample_size=100))

    s = '''
sample_normal = np.random.normal(5, 1, 10000)
test = bootstrap(sample_normal, np.mean, {}, 100)
'''

    setup = '''
import numpy as np
from __main__ import bootstrap
'''

    s_np = '''
sample_normal = np.random.normal(5, 1, 10000)
test = bootstrap_np(sample_normal, np.mean, {}, 100)
'''

    setup_np = '''
import numpy as np
from __main__ import bootstrap_np
'''

    s_par = '''
sample_normal = np.random.normal(5, 1, 10000)
test_p = bootstrap_par(sample_normal, np.mean, {}, 100)
'''

    setup_par = '''
import multiprocessing as mp
import numpy as np
from __main__ import bootstrap_par
'''

    s_np_par = '''
sample_normal = np.random.normal(5, 1, 10000)
test_p = bootstrap_np_par(sample_normal, np.mean, {}, 100)
'''

    setup_np_par = '''
import multiprocessing as mp
import numpy as np
from __main__ import bootstrap_np_par
'''

    instructions_s = [s.format(i) for i in range(1000, 1000000, 10000)]
    instructions_s_par = [s_par.format(i) for i in range(1000, 1000000, 10000)]
    instructions_s_np = [s_np.format(i) for i in range(1000, 1000000, 10000)]
    instructions_s_np_par = [s_np_par.format(i) for i in range(1000, 1000000, 10000)]

    serial = [timeit.Timer(stmt=ins, setup=setup).timeit(1) for ins in instructions_s]
    par = [timeit.Timer(stmt=ins, setup=setup_par).timeit(1) for ins in instructions_s_par]
    np_serial = [timeit.Timer(stmt=ins, setup=setup_np).timeit(1) for ins in instructions_s_np]
    np_par = [timeit.Timer(stmt=ins, setup=setup_np_par).timeit(10) for ins in instructions_s_np_par]
    repeats = range(1000, 1000000, 10000)

    df = pd.DataFrame(np.c_[serial, par, np_serial, np_par], index=repeats,
                      columns=['Serial', 'Parallel', 'SerialNP', 'ParallelNP'])

    sns.set(style='darkgrid', palette='Paired')
    sns.lineplot(data=df, dashes=[(None, None), (2, 2), (None, None), (2, 2)])
    plt.legend(labels=['Serial', 'Parallel', 'Serial with Numpy', 'Parallel with Numpy'])
    plt.show()
