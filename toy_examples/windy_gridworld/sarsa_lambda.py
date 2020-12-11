import sys

sys.path.append('../')

from utilities.profiler import Profiler
import matplotlib.pyplot as plt
import numpy as np
from gridworld import Gridworld, Position
from utilities.utils import argmax_random_tie
import pandas as pd

plt.ion()
plt.show()

rows = 10
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

print_map = True
print_plot = True
update_plot_while_running = True


def run(alpha, eps_start, gamma, lambd, color):
    n_episodes = 100
    print_every = 1
    q = np.zeros((rows, cols, len(moveset)))
    epsilons = np.linspace(eps_start, 0.1, n_episodes)

    steps_to_goal_pr_episode = []
    plot_total_steps = []
    img_world = np.zeros((rows, cols))

    for episode, eps in zip(range(n_episodes), epsilons):
        e = np.zeros((rows, cols, len(moveset)))

        if episode % 1 == 0:
            print("episode {}".format(episode))

        world.reset()
        s = world.pos
        a = np.random.randint(0, len(moveset)) if np.random.uniform() < eps else argmax_random_tie(q[s.row, s.col])
        found = False
        steps = 0
        while not found:
            steps += 1
            world.wind()
            r = world.move(moveset[a])
            sn = world.pos
            an = np.random.randint(0, len(moveset)) if np.random.uniform() < eps else argmax_random_tie(
                q[sn.row, sn.col])
            d = r + gamma * q[sn.row, sn.col, an] - q[s.row, s.col, a]
            e[s.row, s.col, a] += 1

            q += alpha * d * e
            e *= gamma * lambd

            s = Position(sn.row, sn.col)
            a = an

            # Non algorithm:
            if s == world.goal:
                steps_to_goal_pr_episode.append(steps)
                found = True
                plot_total_steps.append(steps)

        if print_map:
            if episode % print_every == 0:
                for r in range(rows):
                    for c in range(cols):
                        block = world.world[r][c]
                        if block == "W":
                            img_world[r][c] = 50
                        elif block == " ":
                            img_world[r][c] = 1
                        else:
                            img_world[r][c] = 100
                plt.figure(1)
                plt.imshow(img_world)
                if update_plot_while_running:
                    plt.pause(0.01)

    if print_plot:
        for w in [n_episodes // 10]:
            plt.figure(2)
            if w == 0:
                plt.plot(range(len(steps_to_goal_pr_episode)), steps_to_goal_pr_episode, "-" + color)
            else:
                rolling = pd.DataFrame(steps_to_goal_pr_episode).rolling(window=w).mean()
                plt.plot(range(len(steps_to_goal_pr_episode)), rolling, "-" + color)
            title = "W = {}, alpha = {:.3f}, eps = {:.3f}, gamma = {:.3f}, lambda = {:.3f}".format(w, alpha, eps, gamma, lambd)
            plt.title(title)
            plt.ylabel("steps to complete")
            plt.xlabel("episode number")
            plt.draw()
            plt.pause(0.001)


alpha = 0.25  # learning-rate
eps_start = 0.1  # random step chance
gamma = 0.01  # discount factor
lambd = 0.3  # if 0 td(0)

# colors = ["b", "g", "r", "c", "m", "y"]
colors = ["b"]
gammas = np.linspace(gamma, gamma, num=len(colors))
alphas = np.linspace(alpha, alpha, num=len(colors))
lambdas = np.linspace(lambd, lambd, num=len(colors))
eps_starts = np.linspace(eps_start, eps_start, num=len(colors))

for eps_start, lambd, gamma, alpha, color in zip(eps_starts, lambdas, gammas, alphas, colors):
    with Profiler():
        run(alpha=alpha, eps_start=eps_start, gamma=gamma, lambd=lambd, color=color)

if print_plot:
    plt.figure(2)
    plt.draw()
    plt.show()
    input("Press [enter] to continue.")
