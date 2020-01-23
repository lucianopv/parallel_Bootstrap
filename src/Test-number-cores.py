import numpy as np
import timeit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src import parallel_bootstrap

s = '''
sample_normal = np.random.normal(5, 1, 10000)
test = parallel_bootstrap.bootstrap(sample_normal, np.mean, 1000, 100)
'''

setup = '''
import numpy as np
from src import parallel_bootstrap
'''

s_np = '''
sample_normal = np.random.normal(5, 1, 10000)
test = parallel_bootstrap.bootstrap_np(sample_normal, np.mean, 1000, 100)
'''

setup_np = '''
import numpy as np
from src import parallel_bootstrap
'''

s_par = '''
sample_normal = np.random.normal(5, 1, 10000)
test_p = parallel_bootstrap.bootstrap_par(sample_normal, np.mean, 1000, 100, {})
'''

setup_par = '''
import multiprocessing as mp
import numpy as np
from src import parallel_bootstrap
'''

s_np_par = '''
sample_normal = np.random.normal(5, 1, 10000)
test_p = parallel_bootstrap.bootstrap_np_par(sample_normal, np.mean, 1000, 100, {})
'''

setup_np_par = '''
import multiprocessing as mp
import numpy as np
from src import parallel_bootstrap
'''


instructions_s_par = [s_par.format(i) for i in range(1, 7, 1)]
instructions_s_np_par = [s_np_par.format(i) for i in range(1, 7, 1)]

serial = [timeit.Timer(stmt=s, setup=setup).timeit(1) for _ in range(1, 7, 1)]
par = [timeit.Timer(stmt=ins, setup=setup_par).timeit(1) for ins in instructions_s_par]
np_serial = [timeit.Timer(stmt=s_np, setup=setup_np).timeit(1) for _ in range(1, 7, 1)]
np_par = [timeit.Timer(stmt=ins, setup=setup_np_par).timeit(1) for ins in instructions_s_np_par]
cores = range(1, 7, 1)

df = pd.DataFrame(np.c_[serial, par, np_serial, np_par], index=cores,
                  columns=['Serial', 'Parallel', 'SerialNP', 'ParallelNP'])

df.to_pickle('../data/Cores.pkl')

df['Cores'] = df.index
df2 = pd.melt(df[:,['Cores', 'Parallel', 'ParallelNP']], ['Cores'])
sns.set(style='darkgrid', palette='Paired')
p = sns.catplot(data=df2, x='Cores', y='value', hue='variable', kind='point', linestyles=["-", "--", "-", "--"],
                legend_out=False)
# p.legend(loc='upper right', ncol=2, labels=['Serial', 'Parallel', 'Serial Numpy', 'Parallel Numpy'])
p.set(xlabel='Cores Used for Parallel', ylabel='Time')
p.fig.text(2.2, 0.9, '*Serial presented for comparison.')
plt.show()
