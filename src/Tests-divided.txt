import numpy as np
import timeit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

setup_sample = '''
import numpy as np
import multiprocessing as mp
from src import parallel_bootstrap_divided
sample_normal = np.random.normal(5, 1, 1000)
'''
setup_statistic = '''
import multiprocessing as mp
import numpy as np
from src import parallel_bootstrap_divided
sample_normal = np.random.normal(5, 1, 1000)
all_samples = [sampling(sample_normal, i, 100) for i in range(1000, 100000, 10000)]
'''

s = '''
parallel_bootstrap_divided.sampling(sample_normal, samples={}, sample_size=100)
'''
s_p = '''
parallel_bootstrap_divided.sampling_par(sample_normal, samples={}, sample_size=100)
'''
s_ps = '''
parallel_bootstrap_divided.sampling_par_shared(sample_normal, samples={}, sample_size=100)
'''
st = '''
parallel_bootstrap_divided.statistic(all_samples[{}], np.mean)
'''
st_p = '''
parallel_bootstrap_divided.statistic_par(all_samples[{}], np.mean)
'''
st_ps = '''
parallel_bootstrap_divided.statistic_par_shared(all_samples[{}], np.mean)
'''
ins_sample = [s.format(i) for i in range(1000, 100000, 10000)]
ins_sample_p = [s_p.format(i) for i in range(1000, 100000, 10000)]
ins_sample_ps = [s_ps.format(i) for i in range(1000, 100000, 10000)]
ins_stat = [st.format(i) for i in range(10)]
ins_stat_p = [st_p.format(i) for i in range(10)]
ins_stat_ps = [st_ps.format(i) for i in range(10)]

sample = [timeit.Timer(stmt=ins, setup=setup_sample).timeit(1) for ins in ins_sample]
sample_p = [timeit.Timer(stmt=ins, setup=setup_sample).timeit(1) for ins in ins_sample_p]
sample_ps = [timeit.Timer(stmt=ins, setup=setup_sample).timeit(1) for ins in ins_sample_ps]
stat = [timeit.Timer(stmt=ins, setup=setup_statistic).timeit(1) for ins in ins_stat]
stat_p = [timeit.Timer(stmt=ins, setup=setup_statistic).timeit(1) for ins in ins_stat_p]
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
