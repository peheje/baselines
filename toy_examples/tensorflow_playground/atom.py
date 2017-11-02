import sys
sys.path.append('../')


from utilities.utils import *
from utilities.profiler import Profiler
import numpy as np
import matplotlib.pyplot as plt

# Collect data

v_min = 0
v_max = 10
n_atoms = 100000
c = n_atoms*10

a = np.zeros(n_atoms)
for i in range(n_atoms):
    if np.random.rand() < 0.5:
        a[i] = np.random.normal(8, 2)
    else:
        a[i] = np.random.normal(2, 1)

#high_value_indices = a > v_max
#a[high_value_indices] = 0

#low_value_indicies = a < v_min
#a[low_value_indicies] = 0

print(probability_chooser(a))

# Histogram
histogram_from_atoms(a)