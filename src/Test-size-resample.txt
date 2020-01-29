import numpy as np
import timeit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

s = '''
sample_normal = np.random.normal(5, 1, 10000)
test = parallel_bootstrap.bootstrap(sample_normal, np.mean, {}, 100)
'''

setup = '''
import numpy as np
from src import parallel_bootstrap
'''

s_np = '''
sample_normal = np.random.normal(5, 1, 10000)
test = parallel_bootstrap.bootstrap_np(sample_normal, np.mean, {}, 100)
'''

setup_np = '''
import numpy as np
from src import parallel_bootstrap
'''

s_par = '''
sample_normal = np.random.normal(5, 1, 10000)
test_p = parallel_bootstrap.bootstrap_par(sample_normal, np.mean, {}, 100)
'''

setup_par = '''
import multiprocessing as mp
import numpy as np
from src import parallel_bootstrap
'''

s_np_par = '''
sample_normal = np.random.normal(5, 1, 10000)
test_p = parallel_bootstrap.bootstrap_np_par(sample_normal, np.mean, {}, 100)
'''

setup_np_par = '''
import multiprocessing as mp
import numpy as np
from src import parallel_bootstrap
'''

instructions_s = [s.format(i) for i in range(10, 9000, 100)]
instructions_s_par = [s_par.format(i) for i in range(10, 9000, 100)]
instructions_s_np = [s_np.format(i) for i in range(10, 9000, 100)]
instructions_s_np_par = [s_np_par.format(i) for i in range(10, 9000, 100)]

serial = [timeit.Timer(stmt=ins, setup=setup).timeit(1) for ins in instructions_s]
par = [timeit.Timer(stmt=ins, setup=setup_par).timeit(1) for ins in instructions_s_par]
np_serial = [timeit.Timer(stmt=ins, setup=setup_np).timeit(1) for ins in instructions_s_np]
np_par = [timeit.Timer(stmt=ins, setup=setup_np_par).timeit(1) for ins in instructions_s_np_par]
repeats = range(10, 9000, 100)

df = pd.DataFrame(np.c_[serial, par, np_serial, np_par], index=repeats,
                  columns=['Serial', 'Parallel', 'SerialNP', 'ParallelNP'])

df.to_pickle('../data/NoNPvsNP_small.pkl')

sns.set(style='darkgrid', palette='Paired')
sns.lineplot(data=df, dashes=[(None, None), (2, 2), (None, None), (2, 2)])
plt.legend(labels=['Serial', 'Parallel', 'Serial with Numpy', 'Parallel with Numpy'])
plt.show()
