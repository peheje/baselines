import sys

sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from utilities.profiler import Profiler
from collections import Counter


def amax_probability_chooser(arr):
    # find max index of arr by calling probability_chooser for all
    values = np.empty(len(arr))
    for i, atoms in enumerate(arr):
        values[i] = probability_chooser(atoms)
    return argmax_random_tie(values)


def probability_chooser(atoms):
    hist, bin_edges = np.histogram(atoms, bins=10)
    return bin_edges[np.argmax(hist)]


def histogram_from_atoms(atoms):


    n, bins, patches = plt.hist(atoms, bins=100)
    plt.xlabel('Reward')
    plt.ylabel('Probability')
    plt.axis()
    plt.grid(True)
    plt.show()


def argmax_random_tie(arr):
    return np.random.choice(np.where(arr == arr.max())[0])


def argmax_random_tie2(arr):
    """ Only faster if len(arr) < 8 """
    m = arr.max()
    l = len(arr)
    start_i = np.random.randint(0, l)
    for i in range(start_i, l + start_i):
        idx = i % l
        if arr[idx] == m:
            return idx


if __name__ == "__main__":
    arr = np.random.random(10)
    arr2 = np.copy(arr)
    iters = 100000

    r1 = []
    r2 = []

    with Profiler():
        for n in range(iters):
            np.random.seed(42)
            r1.append(argmax_random_tie(arr))

    with Profiler():
        for n in range(iters):
            np.random.seed(42)
            r2.append(argmax_random_tie2(arr2))

    assert np.array_equal(r1, r2)
