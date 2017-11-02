import time
from copy import copy

import numpy as np

from gridworld import Gridworld, Position
from utilities.utils import argmax_random_tie

# START
rows = 12
cols = 12
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
print_every = 1000
q = np.zeros((rows, cols, len(moveset)))

plot_xs_episode = []
plot_ys_steps = []

print(world)

for episode in range(n_episodes):
    print("episode {}".format(episode))
    world.reset()
    s = world.pos
    a = np.random.randint(0, len(moveset)) if np.random.uniform() < e else argmax_random_tie(q[s.row][s.col])
    found = False
    e *= e_decay
    steps_completion = 0
    while not found:
        steps_completion += 1
        #world.wind()
        r = world.move(moveset[a])
        sn = world.pos
        an = np.random.randint(0, len(moveset)) if np.random.uniform() < e else argmax_random_tie(q[sn.row][sn.col])
        q[s.row][s.col][a] += alpha * (r + lambd * q[sn.row][sn.col][an] - q[s.row][s.col][a])
        s = copy(sn)
        a = an

        if s == world.goal:
            found = True
            plot_xs_episode.append(episode)
            plot_ys_steps.append(steps_completion)

        if episode % print_every == 0 and steps_completion < 30:
            print(world)
            time.sleep(0.1)
            if found:
                print("found goal, epsilon {} in steps {}".format(e, steps_completion))
                time.sleep(3)

# plt.xlabel("Episodes")
# plt.ylabel("Steps for completion")
# xs_mean, ys_mean = utilities.moving_avg(plot_ys_steps, 3)
# plt.plot(xs_mean, ys_mean, "-r")
# plt.axis([0, n_episodes, 0, 100])
# plt.show()
