

import pandas as pd
df = pd.DataFrame([['a', 'b'], ['c', 'd']],
                   index=['row 1', 'row 2'],
                  columns=['col 1', 'col 2'])

a = df.to_json(orient = "index")

tmp = pd.read_json(a, orient = "index")


# rpy2 instalieren über miniconda (version mit python 3.6: Miniconda3-4.3.31-Windows-x86_64): conda install -c r rpy2
import rpy2
print(rpy2.__version__)

import rpy2.robjects as robjects

from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')

pi = robjects.r['pi']
pi[0]