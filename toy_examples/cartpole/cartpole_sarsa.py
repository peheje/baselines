import sys
sys.path.append('../')
from time import time
import gym
import numpy as np
from utilities.utils import argmax_random_tie


def process_state(s):
    ranges = [[-2.4, 2.4], [-10, 10], [-0.3, 0.3], [-10, 10]]
    s_discretized = np.empty(len(ranges))
    for i, r in enumerate(ranges):
        s_discretized[i] = np.interp(s[i], r, [0, res - 1])
        s_discretized[i] = int(s_discretized[i])
    return np.array(s_discretized, dtype=np.int)


env = gym.make('CartPole-v0')
env.reset()


n_episodes = 10_00
alpha = 0.2  # learning-rate
eps_start = 0.8  # random step chance
eps_end = 0.05
epsilons = np.linspace(eps_start, eps_end, n_episodes)
gamma = 0.7  # discount factor
lambd = 0.9  # if 0 td(0)
e_decay = 0.999  # epsilon decay each episode

print_every = 100
res = 20
q = np.zeros((res, res, res, res, 2))

train = False

# while True:
#     s, r, done, info = env.step(0)
#     print(s)
#     print(process_state(s))
#     sleep(1)
#     env.render()
#     if done:
#         print("obs: {}, rew: {}, done: {}, info: {}".format(s, r, done, info))

if not train:
        print("loading q from file")
        epsilons = np.zeros(n_episodes)
        q = np.load("/tmp/cartpole_q.npy")

for episode in range(n_episodes):
    eps = epsilons[episode]
    e = np.zeros((res, res, res, res, 2))
    x1, y1, x2, y2 = process_state(env.reset())
    a = np.random.randint(0, 2) if np.random.uniform() < eps else argmax_random_tie(q[x1, y1, x2, y2])
    episode_done = False
    eps_start *= e_decay
    steps = 0
    if episode % print_every == 0:
        print("episode {}, epsilon {}".format(episode, eps))
    while not episode_done:
        steps += 1
        sn, r, done, info = env.step(a)
        cart_pos, pole_a = sn[0], sn[3]

        x1n, y1n, x2n, y2n = process_state(sn)
        an = np.random.randint(0, 2) if np.random.uniform() < eps else argmax_random_tie(q[x1n, y1n, x2n, y2n])

        d = r + gamma * q[x1n, y1n, x2n, y2n, an] - q[x1, y1, x2, y2, a]
        e[x1, y1, x2, y2, a] += 1

        q += alpha * d * e
        e *= gamma * lambd

        x1, y1, x2, y2 = x1n, y1n, x2n, y2n
        a = an

        if done:
            episode_done = True
            if steps > 199:
                print("{} : done steps {}".format(time(), steps))
                # print(x1, y1, x2, y2)
                # print("steps: {}".format(steps))        

        if not train:
            env.render()

if train:
    print("saving model")
    np.save("/tmp/cartpole_q", q)
