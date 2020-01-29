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
# which will allow it to be shared by Processes during parallelization

# We compare three versions of the function for both parts:
# 1) serial form
# 2) parallelized form with numpy array input
# 3) parallelized form with shared array input

import numpy as np
import multiprocessing as mp
import ctypes
from typing import Callable


def sampling(x, samples=1000, sample_size=100):
    """
    This function generates a determined number of samples from an initial array. 
    It uses the numpy library to randomly sample from the initial array.

    Parameters
    ----------
    x : np.ndarray
        The array of data to which the sampling process is applied
    samples : int
        Number of samples to take from x
    sample_size :   int
        Size of each sample, should be between 0 and the total length of x

    Returns
    -------
    np.ndarray 
        An array of all samples taken from x, where each sample is an array
        of size sample_size.
        
    Examples
    --------
    Given an array of data taken from a normal distribution(5, 1), take 1000 samples
    of size 100 from the data.
    >>> sample_normal = np.random.normal(5, 1, 1000)
    >>> sampling(sample_normal, samples = 1000, sample_size = 100)
    """
    return np.array([np.random.choice(x, sample_size) for _ in range(samples)])


def sampling_par(x, samples=1000, sample_size=100):
    """
    This function generates a determined number of samples from an initial array. 
    It uses the numpy library to randomly sample from the initial array. Using the 
    multiprocessing package it parallelises the process by creating a 
    pool of workers and dividing up the sampling tasks between them. The number of 
    workers is the same as the number of CPUs in the computer.
    
    Parameters
    ----------
    x : np.ndarray
        The array of data to which the sampling process is applied
    samples : int
        Number of samples to take from x
    sample_size :   int
        Size of each sample, should be between 0 and the total length of x

    Returns
    -------
    np.ndarray 
        An array of all samples taken from x, where each sample is an array 
        of size sample_size.
        
    Examples
    --------
    Given an array of data taken from a normal distribution(5, 1), take 1000 samples
    of size 100 from the data.
    >>> sample_normal = np.random.normal(5, 1, 1000)
    >>> sampling_par(sample_normal, samples = 1000, sample_size = 100)
    """
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap_async(np.random.choice, [(x, sample_size) for _ in range(samples)]).get()
    pool.close()
    return results


def sampling_par_shared(x, samples=1000, sample_size=100):
    """
    This function generates a determined number of samples from an initial array. 
    It uses the numpy library to randomly sample from the initial array. Using the 
    multiprocessing package it parallelises the process by creating a 
    pool of workers and dividing up the sampling tasks between them. Before parallelisation, 
    the initial numpy array is turned into a memory shared array, meaning that the 
    array can be shared by the workers and avoids creating multiple copies of it. 
    The number of workers is the same as the number of CPUs in the computer.
    
    Parameters
    ----------
    x : np.ndarray
        The array of data to which the sampling process is applied
    samples : int
        Number of samples to take from x
    sample_size :   int
        Size of each sample, should be between 0 and the total length of x

    Returns
    -------
    np.ndarray 
        An array of all samples taken from x, where each sample is an array 
        of size sample_size.

    Examples
    --------
    Given an array of data taken from a normal distribution(5, 1), take 1000 samples
    of size 100 from the data.
    >>> sample_normal = np.random.normal(5, 1, 1000)
    >>> sampling_par_shared(sample_normal, samples = 1000, sample_size = 100)
    """
    x_ = mp.Array(ctypes.c_double, len(x))
    x_shr = np.ctypeslib.as_array(x_.get_obj())
    x_shr[:] = x

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap_async(np.random.choice, [(x, sample_size) for _ in range(samples)]).get()
    pool.close()
    return results


def statistic(x, func):
    """
    This function applies a determined function to each sample in the input array of samples.

    Parameters
    ----------
    x : np.ndarray
        The array of samples, where each sample is an array, for which 
        the statistic function is applied to each sample
    func : function
        The statistical function to be applied to each sample in x
        
    Returns
    -------
    np.ndarray 
        An array of outputs from the function applied to each sample

    Examples
    --------
    Given an array of data taken from a normal distribution(5, 1), take 1000 samples
    of size 100 from the data. Given this array of samples, get the mean for each sample.
    >>> sample_normal = np.random.normal(5, 1, 1000)
    >>> samples = sampling(sample_normal, samples = 1000, sample_size = 100)
    >>> sample_means = statistic(samples, func = np.mean)
    """
    return np.array([func(z) for z in x])


def statistic_par(x, func):
    """
    This function applies a determined function to each sample in the input array of samples.
    Using the multiprocessing package it parallelises the process by creating a pool of 
    workers and dividing up the tasks between them. The number of workers is the same as
    the number of CPUs in the computer.

    Parameters
    ----------
    x : np.ndarray
        The array of samples, where each sample is an array, for which 
        the statistic function is applied to each sample
    func : function
        The statistical function to be applied to each sample in x
        
    Returns
    -------
    np.ndarray 
        An array of outputs from the function applied to each sample

    Examples
    --------
    Given an array of data taken from a normal distribution(5, 1), take 1000 samples
    of size 100 from the data. Given this array of samples, get the mean for each sample.
    >>> sample_normal = np.random.normal(5, 1, 1000)
    >>> samples = sampling(sample_normal, samples = 1000, sample_size = 100)
    >>> sample_means = statistic(samples, func = np.mean)
    """
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map_async(func, x).get()
    pool.close()
    return results


def statistic_par_shared(x: np.ndarray, func: Callable):
    """
    This function applies a determined function to each sample in the input array of samples.
    Using the multiprocessing package it parallelises the process by creating a pool of 
    workers and dividing up the tasks between them. Before parallelisation, the initial
    numpy array is turned into a memory shared array, meaning that the array can be 
    shared by the workers and avoids creating multiple copies of it. The number of workers 
    is the same as the number of CPUs in the computer.

    Parameters
    ----------
    x : np.ndarray
        The array of samples, where each sample is an array, for which 
        the statistic function is applied to each sample
    func : function
        The statistical function to be applied to each sample in x
        
    Returns
    -------
    np.ndarray 
        An array of outputs from the function applied to each sample

    Examples
    --------
    Given an array of data taken from a normal distribution(5, 1), take 1000 samples
    of size 100 from the data. Given this array of samples, get the mean for each sample.
    >>> sample_normal = np.random.normal(5, 1, 1000)
    >>> samples = sampling(sample_normal, samples = 1000, sample_size = 100)
    >>> sample_means = statistic(samples, func = np.mean)
    """
    x_ = [mp.Array(ctypes.c_double, len(x)) for _ in range(x.shape[1])]
    x_shr = [np.ctypeslib.as_array(z.get_obj()) for z in x_]
    x_shr[:] = x

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map_async(func, x_shr).get()

    pool.close()
    return results