# This is the modified version of "parallel_bootstrap_div.py"
'''
To make our testing faster and code clearer, I made the following modifications to "parallel_bootstrap_div.py":
    
    1. I changed the "naming", (e.g. x -> sample, samples -> num_resamples, sample_size -> resample_size)
        sample: 1D array, the sample that we are going to bootstrap (resample)
        num_resamples: the number of resamples / how many times do we resample the sample
        resample_size: the size of each resample (all resamples should have equal size)
        
        For the purpose of running time complexity analysis,
        we denote "num_resamples" as 'm' and "resample_size" as 'n'.
    
    2. I replaced "instructions" with a simpler implemetation.
    
    3. Instead of using "timeit.Timer" class, I defined new methods for timing.
    
    4. In our old implementation, the processes of setup (i.e. importing all kinds of stuff) 
       and generating the sample are all timed in the Timer class.
       I do not think this is necessary, because:
           1. these processes are repeated many times. 
           2. most of these processes are the same in the serial and parallel version (divided).
           3. timing these processes will only slow our overall program.
       I think is better to only time the process of applying statistic function to resamples,
       which is exactly what we want to test in the "divided" scenario.        
       Therefore, I minimized the scope of our timing methods to 
       only testing the process of applying statistic function to resamples.
       
    If you compare the results from this version with the results from the old version,
    (with "num_resamples" and "resample_size" set to the same value in both version)
    you can see that the overall running time of the program is greatly reduced!
    This boosts our efficiency if we wanna do more tests and larger tests afterwards!


    Running time comparison analysis is attached at the BOTTOM of this code.
'''

from typing import Callable
import time
import numpy as np
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Resamples a sample with replacement "num_resamples" times
# Returns a 2D array which has shape "num_resamples" x "resample_size"
def resample(sample, num_resamples, resample_size):
    return np.array([np.random.choice(sample, resample_size) for _ in range(num_resamples)])


# This function applies the statistic function to each resample using serial computing
# As we do not care about the value of statistic estimator, this function only returns the running time
def runtime_serial(resamples: np.ndarray, func: Callable) -> float:
    t_start =  time.perf_counter()
    for resample in resamples:     # m steps
        func(resample)             # n steps
                                   # running time is O(mn)
    t_end = time.perf_counter()
    return t_end - t_start


# This function applies the statistic function to each resample using parallel computing
# As we do not care about the value of statistic estimator, this function only returns the running time
def runtime_parallel(resamples: np.ndarray, func: Callable, pool: mp.pool) -> float:
    t_start =  time.perf_counter()
    pool.map(func, resamples, 500) # running time should be O(mn)
    t_end = time.perf_counter()
    return t_end - t_start


if __name__ == "__main__":
    '''
    In our tests, the size of sample and "resample_size" are both set to 100 by default.
    We start with 1000 resamples and increment "num_resamples" by 10000 each repetition.
    To speed up my personal testing, I set the max "num_resamples" to 100000,
    you guys can change it to 1000000 or even larger when you are ready to pickle the data.
    '''
    sample = np.random.normal(10, 2, 100)
    repeats = range(1000, 100000, 10000)
    
    times_serial = [runtime_serial(resample(sample, i, 100), np.mean) for i in repeats]
    
    with mp.Pool(processes = mp.cpu_count()) as pool:
        times_parallel = [runtime_parallel(resample(sample, i, 100), np.mean, pool) for i in repeats]
    
    data = pd.DataFrame(np.c_[times_serial, times_parallel], index = repeats, columns = ['Serial', 'Parallel'])
    
    sns.set(style='darkgrid', palette='Paired')
    sns.lineplot(data = data, dashes = [(None, None), (2, 2)])
    plt.legend(labels=['Serial', 'Parallel'])
    plt.show()
    print(data)
    
'''
Some thoughts on the comparison of running time:

Two observations can be drawn:
1. There is a spike at the beginning of the dotted line (i.e. parallel).
2. Parallel computing takes slightly more time than serial computing in this scenario.

Before I present my thoughts, I need to go through what Luciano and Lora have suggested so far regarding this issue:
    
    Luciano: In the "Using parallel only on simple functions" section of file "FinalVersionBootstrapping.ipynb", 
             Luciano pointed out:
             "From the graph one can see that both algorithm have a time complexity of O(n)
              but the Parallel implementation takes longer to accomplish the task. 
              When one compares the serial and the parallel function, 
              one can see that the serial just takes one line of code to accomplish the task, 
              although the this line has a time complexity of O(n)
              Meanwhile, the parallel version uses additional steps to create the pool object and the processes, 
              therefore the preparation to apply the function of the statistic 
              takes more time than the serial implementation."
              
             My opinions: If we treat the size of each resample as a constant, 
             then both algorithm indeed have a time complexity of O(n), where n is the number of resamples.
             If we treat the size of each resample (i.e "resample_size") as a variable,
             then both algorithm have a time complexity of O(mn),
             where m is the number of resamples, and n is the size of each resample.
             The fact that additional steps are needed to create the pool object cannot explain the second observation,
             because I created only one pool object in this simplified code and we still get the second observation.
             However, I think your argument may help explain the first observation, which I will explain later.
             
    Lora:    In your email, you listed a few potential explanations to the second observation:
             Time cost added by parallelization:
             "Pickling and unpickling of data", "distribution of tasks between processors"
             Other factors:
             "Windows, Spyder, Jupyter may all have an effect on how multiprocessing performs" 
             
             My opinions: I do not understand why you put "pickling" here as we did not time it.
             I agree with you on all other factors you mentioned.
             "Distribution of tasks" and "Softwares" all have an effect on the performance of parallelization.
             I think there is a technical term for this ----- Parallelization Overhead (PO).
             Check out the link I sent to the group, that guy explains this well.
             These factors can lead to the second observation.
    
First observation:

I do not have a confident explanation for the first observation. Further research is needed.
However, one possible explanations can be given:
"The parallel version uses additional steps to create the pool object".
Although this argument cannot explain the second observation, it gives some clues to the first observation.
Line 84 seems to create a new pool obeject with 4 processes using "with...as..." statement, 
but the creation of pool object may actually not happen when this line is executed. (further research is needed)
My guess is, the creation of pool object happens when the pool object is referenced at the first time,
and slows down the first reference to the pool object,
which is exactly when the "spike" happens. Once the pool object is created, 
future reference to the pool object will not be slowed down, 
which is why the spike only shows up at the beginning.
(Line 67 references the pool obeject, and this line is executed multiple times)


Second observation:   

Reason why parallel computation runs slower than serial computation in this case:

The weight of PO increases with shorter absolute computation time per taskel.

Definition on PO and "taskel" can be found in the link I sent to the group.

More specifically,
The amount of time reduced by switching from serial to parallel computing is smaller than the amount of time increased by PO.
Therefore, the benefit of parallel computing is not seen in this particular scenario, instead, we see the time cost caused by parallelization overhead.
    
"The tasks need to be computationally heavy (intensive) enough, to earn back the PO we have to pay for parallelization. 
The relevance of PO decreases with increasing absolute computation time per taskel. 
Or, to put it the other way around, the bigger the absolute computation time per taskel for your problem, 
the less relevant gets the need for reducing PO. If your computation will take hours per taskel, 
the IPC overhead will be negligible in comparison."  -----cited from the link I sent to the group

To generalize, one can predict that:
When the absolute computation time per taskel is very short, there will be no need to use parallel computing,
because the negative impact of PO will be significant in comparison.
However, when the absolute computation time per taskel is long, we can speed up the program by parallel computing,
because the negative effect of PO will become negligible in comparison.
'''
    

