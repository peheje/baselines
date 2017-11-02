import time
from copy import copy

import numpy as np

from gridworld import Gridworld, Position
from utilities.utils import argmax_random_tie

# START
rows = 7
cols = 10
world = Gridworld(rows, cols)
moveset = [
    # Position(1, 1),
    # Position(1, -1),
    # Position(-1, -1),
    # Position(-1, 1),
    Position(1, 0),  # down
    Position(0, -1),  # left
    Position(0, 1),  # right
    Position(-1, 0)  # up
]

alpha = 0.2
e = 0.3
lambd = 0.2
e_decay = 0.9999
n_episodes = 100000
print_every = 10000
q = np.zeros((rows, cols, len(moveset)))

for episode in range(n_episodes):
    print("episode {}".format(episode))
    world.reset()
    s = world.pos
    found = False
    e *= e_decay
    step = 0

    while not found:
        step += 1
        world.wind()
        a = np.random.randint(0, len(moveset)) if np.random.uniform() < e else argmax_random_tie(q[s.row][s.col])
        r = world.move(moveset[a])
        sn = world.pos
        q[s.row][s.col][a] += alpha * (r + lambd * np.max(q[sn.row][sn.col]) - q[s.row][s.col][a])
        s = copy(sn)

        if s == world.goal:
            found = True

        if episode % print_every == 0 and step < 30:
            print(world)
            time.sleep(0.1)
            if found:
                print("found goal, epsilon {} in steps {}".format(e, step))
                time.sleep(3)
