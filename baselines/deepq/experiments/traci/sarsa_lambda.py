import sys

sys.path.append('../')
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from utilities.utils import argmax_random_tie
import gym

import TraciSimpleEnv.TraciSimpleEnv

env = gym.make('TraciSimpleEnv-v0')

alpha = 0.2  # learning-rate
eps = 0.3  # random step chance
gamma = 0.9  # discount factor
lambd = 0.9  # if 0 td(0)
e_decay = 0.9995  # epsilon decay each episode

n_episodes = 100000
print_every = 10

n_actions = 3
n_states = env.max_cars_in_queue+1
state_actions = (n_states, n_states, n_states, n_states, n_actions)

q = np.zeros(state_actions)

plot_total_steps = []
plot_total_reward = []

for episode in range(n_episodes):
    e = np.zeros((state_actions))
    print("episode {}".format(episode))

    state = env.reset()
    action = np.random.randint(0, n_actions) if np.random.uniform() < eps else argmax_random_tie(q[tuple(state)])
    found = False
    steps = 0
    totalReward = 0
    while not found:
        eps *= e_decay

        steps += 1

        state_next, reward, done, info = env.step(action)

        action_next = np.random.randint(0, n_actions) if np.random.uniform() < eps else argmax_random_tie(q[tuple(state_next)])
        d = reward + gamma * q[tuple(state_next)][action_next] - q[tuple(state)][action]
        e[tuple(state)][action] += 1

        q += alpha * d * e
        e *= gamma * lambd

        state = copy(state_next)
        action = action_next
        totalReward += reward

        if steps % print_every == 0:
            print("epsilon", eps)
            print("reward", reward)
            print("state", state)
            # print("q", q)

        if done:
            found = True
            plot_total_steps.append(steps)
            plot_total_reward.append(totalReward)
            # print("found goal, epsilon {} in steps {}".format(eps, steps))
            # time.sleep(3)

            # if episode % print_every == 0 and steps < 200:
            # print(world)

            # if episode % print_every == 0:
            #     for r in range(rows):
            #         for c in range(cols):
            #             imworld[r][c] = 1 if world.world[r][c] == " " else 0
            #     plt.imshow(imworld)
            #     plt.pause(0.05)

plt.xlabel("Episode")
plt.ylabel("Steps")
plt.plot(range(len(plot_total_steps)), plot_total_steps, "-r")
plt.show()

plt.xlabel("Episode")
plt.ylabel("total_reward")
plt.plot(range(len(plot_total_reward)), plot_total_reward, "-r")
plt.show()
