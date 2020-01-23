# This is the modified version of "parallel_bootstrap.py"
from typing import Callable
import numpy as np
import multiprocessing as mp
import random
import ctypes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time


def bootstrap(sample: np.ndarray, func: Callable, resample_size: int = None) -> float:  
    rindex = [random.randint(0, len(sample) - 1) for _ in range(resample_size)]
    func([sample[i] for i in rindex])


def bootstrap_np(sample: np.ndarray, func: Callable, resample_size: int = None) -> float:
    func([np.random.choice(sample, resample_size)])


def runtime_serial(sample: np.ndarray, func: Callable, num_resamples: int, resample_size: int = None) -> float:
    assert isinstance(sample, np.ndarray), "Please convert the input into an numpy ndarray"
    sample_size = len(sample)
    if resample_size == None or resample_size > sample_size:
        resample_size = sample_size   
    t_start = time.perf_counter()
        
    for _ in range(num_resamples):
        bootstrap(sample, func, resample_size)
   
    t_end = time.perf_counter()
    return t_end - t_start


def runtime_serial_np(sample: np.ndarray, func: Callable, num_resamples: int, resample_size: int = None) -> float:   
    assert isinstance(sample, np.ndarray), "Please convert the input into an numpy ndarray"
    sample_size = len(sample)
    if resample_size == None or resample_size > sample_size:
        resample_size = sample_size        
    t_start = time.perf_counter()
    
    for _ in range(num_resamples):
        bootstrap_np(sample, func, resample_size)
    
    t_end = time.perf_counter()    
    return t_end - t_start
    

def runtime_parallel(pool: mp.pool, sample: np.ndarray, func: Callable, num_resamples: int, resample_size: int = None) -> float: 
    assert isinstance(sample, np.ndarray), "Please convert the input into an numpy ndarray"
    sample_size = len(sample)
    if resample_size == None or resample_size > sample_size:
        resample_size = sample_size
    sample_ = mp.Array(ctypes.c_double, sample_size)
    sample_shr = np.ctypeslib.as_array(sample_.get_obj())
    sample_shr[:]= sample    
    t_start = time.perf_counter()
    
    pool.starmap_async(bootstrap, [(sample_shr, func, resample_size) for _ in range(num_resamples)]).get()
    
    t_end = time.perf_counter()    
    return t_end - t_start


def runtime_parallel_np(pool: mp.pool, sample: np.ndarray, func: Callable, num_resamples: int, resample_size: int = None) -> float: 
    assert isinstance(sample, np.ndarray), "Please convert the input into an numpy ndarray"
    sample_size = len(sample)
    if resample_size == None or resample_size > sample_size:
        resample_size = sample_size
    sample_ = mp.Array(ctypes.c_double, sample_size)
    sample_shr = np.ctypeslib.as_array(sample_.get_obj())
    sample_shr[:]= sample  
    t_start = time.perf_counter()
    
    pool.starmap_async(bootstrap_np, [(sample_shr, func, resample_size) for _ in range(num_resamples)]).get()
    
    t_end = time.perf_counter()  
    return t_end - t_start


if __name__ == '__main__':
    
    repeats = range(1000, 50000, 10000)
    sample = np.random.normal(10, 2, 100)
    
    times_serial = [runtime_serial(sample, np.mean, i) for i in repeats]
    times_serial_np = [runtime_serial_np(sample, np.mean, i) for i in repeats]
    with mp.Pool(mp.cpu_count()) as pool:
        times_parallel = [runtime_parallel(pool, sample, np.mean, i) for i in repeats]
        times_parallel_np = [runtime_parallel_np(pool, sample, np.mean, i) for i in repeats]
    
    df = pd.DataFrame(np.c_[times_serial, times_parallel, times_serial_np, times_parallel_np], index=repeats, 
                      columns=['Serial', 'Parallel', 'SerialNP', 'ParallelNP'])
    
    sns.set(style='darkgrid', palette='Paired')
    sns.lineplot(data=df, dashes=[(None, None), (2, 2), (None, None), (2, 2)])
    plt.legend(labels=['Serial', 'Parallel', 'Serial with Numpy', 'Parallel with Numpy'])
    plt.show()
    print(df)