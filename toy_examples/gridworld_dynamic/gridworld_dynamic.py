import copy
import numpy as np
import tabulate

# Given any position in world, how many steps in average before we reach 1.
# We move N,S,E,W with random (0.25) chance.
# Moving one step in any direction costs -1 if we're not already standing on 1.
# Moving out of bounds still costs -1 but we stay stationary.

cost = -1

world = [
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]
]

v = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

rows = len(v)
cols = len(v[0])
N = 1000

for n in range(N):
    v_next = copy.deepcopy(v)
    for r in range(rows):
        for c in range(cols):
            if world[r][c] != 1:
                east = v[r + 1][c] + cost if r + 1 < rows else v[r][c] + cost
                west = v[r - 1][c] + cost if r - 1 > -1 else v[r][c] + cost
                north = v[r][c - 1] + cost if c - 1 > -1 else v[r][c] + cost
                south = v[r][c + 1] + cost if c + 1 < cols else v[r][c] + cost
                v_next[r][c] = np.mean([east, west, north, south])
    v = v_next
    #print(tabulate.tabulate(v))

print("Iterations: " + str(N))
print(tabulate.tabulate(v))
