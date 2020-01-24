# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 18:35:07 2020

@author: loram
"""
# This file divides the problem into its two parts: 
# 1) sampling n times from the original data (in this example, sample_normal)
# 2) calculating the statistic (in this example, the mean) for each sample
# Both parts can be achieved by applying the function in a serial or parallel form

# When applying the function in a parallel form, it can be applying directly to 
# the input array OR the input array can first be transformed into a shared array
# which will allow it to be shared by Processes during parallelization ??? 
# (not actually sure whether that is what happens - need to read more on this)

# We compare three versions of the function for both parts:
# 1) serial form
# 2) parallelized form with numpy array input
# 3) parallelized form with shared array input

import numpy as np
import multiprocessing as mp
import ctypes
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, List

def sampling(x, samples=1000, sample_size=100):
    return np.array([np.random.choice(x, sample_size) for _ in range(samples)])

def sampling_par(x, samples=1000, sample_size=100):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap_async(np.random.choice, [(x, sample_size) for _ in range(samples)]).get()
    pool.close()
    return results

def sampling_par_shared(x, samples=1000, sample_size=100):
    x_ = mp.Array(ctypes.c_double, len(x))
    x_shr = np.ctypeslib.as_array(x_.get_obj())
    x_shr[:] = x

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap_async(np.random.choice, [(x, sample_size) for _ in range(samples)]).get()
    pool.close()
    return results

#takes list of "samples" samples of sample size "sample_size" and applies the function "func" to each sample
def statistic(x, func):
    return np.array([func(z) for z in x])

#takes list of "samples" samples of sample size "sample_size" and applies the function "func" to each sample
def statistic_par(x, func):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map_async(func, x).get()
    pool.close()
    return results


def statistic_par_shared(x: np.ndarray, func: Callable):
    x_ = [mp.Array(ctypes.c_double, len(x)) for i in range(x.shape[1])]
    x_shr = [np.ctypeslib.as_array(z.get_obj()) for z in x_]
    x_shr[:] = x

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map_async(func, x_shr).get()

    pool.close()
    return results

if __name__ == "__main__":
    # sample_normal = np.random.normal(5, 1, 1000)
    # result = sampling(sample_normal, 1000, 100)
    # print(bootstrap(result, np.mean)[:5])

    setup_sample = '''
import numpy as np
import multiprocessing as mp
from __main__ import sampling, sampling_par, sampling_par_shared
sample_normal = np.random.normal(5, 1, 1000)
'''
    setup_statistic = '''
import multiprocessing as mp
import numpy as np
from __main__ import sampling, statistic, statistic_par, statistic_par_shared
sample_normal = np.random.normal(5, 1, 1000)
all_samples = [sampling(sample_normal, i, 100) for i in range(1000, 100000, 10000)]
'''

    s = '''
sampling(sample_normal, samples={}, sample_size=100)
'''
    s_p = '''
sampling_par(sample_normal, samples={}, sample_size=100)
'''    
    s_ps = '''
sampling_par_shared(sample_normal, samples={}, sample_size=100)
'''
    st = '''
statistic(all_samples[{}], np.mean)
'''
    st_p = '''
statistic_par(all_samples[{}], np.mean)
'''    
    st_ps = '''
statistic_par_shared(all_samples[{}], np.mean)
'''
    ins_sample = [s.format(i) for i in range(1000, 100000, 10000)]
    ins_sample_p = [s_p.format(i) for i in range(1000, 100000, 10000)]
    ins_sample_ps = [s_ps.format(i) for i in range(1000, 100000, 10000)]
    ins_stat = [st.format(i) for i in range(10)]
    ins_stat_p = [st_p.format(i) for i in range(10)]
    ins_stat_ps = [st_ps.format(i) for i in range(10)]

    #print("next")
    sample = [timeit.Timer(stmt=ins, setup=setup_sample).timeit(1) for ins in ins_sample]
    #print("next")
    sample_p = [timeit.Timer(stmt=ins, setup=setup_sample).timeit(1) for ins in ins_sample_p]
    #print("next")
    sample_ps = [timeit.Timer(stmt=ins, setup=setup_sample).timeit(1) for ins in ins_sample_ps]
    #print("next")
    stat = [timeit.Timer(stmt=ins, setup=setup_statistic).timeit(1) for ins in ins_stat]
    #print("next")
    stat_p = [timeit.Timer(stmt=ins, setup=setup_statistic).timeit(1) for ins in ins_stat_p]
    #print("next")
    stat_ps = [timeit.Timer(stmt=ins, setup=setup_statistic).timeit(1) for ins in ins_stat_ps]

    repeats = range(1000, 100000, 10000)

    df = pd.DataFrame(np.c_[sample, sample_p, sample_ps, stat, stat_p, stat_ps], 
                      index=repeats, columns=['Sampling', 'SamplingP', 'SamplingPS', 
                                              'Statistic', 'StatisticP', 'StatisticPS'])

    df.to_pickle('../data/DivParv2.pkl')

    sns.set(style='darkgrid', palette='Paired')
    sns.lineplot(data=df)
    plt.legend(labels=['Sampling', 'SamplingP', 'SamplingPS', 
                                              'Statistic', 'StatisticP', 'StatisticPS'])
    plt.show()
