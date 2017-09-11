import gym
import numpy as np
from baselines import deepq
import TraciSimpleEnv.TraciSimpleEnv

n_episodes = 10000000


def longest_queue_action(state, old_action):
    print(state)
    cars_ns = state[1] + state[3]
    cars_we = state[0] + state[2]
    if cars_ns == cars_we:
        print("same amount, keep light as is")
        return old_action
    elif cars_ns > cars_we:
        print("most cars from north or south")
        return 0
    else:
        print("most cars from west or east")
        return 1


def main():
    env = gym.make('TraciSimpleEnv-v0')
    s = env.reset()
    a = 0

    for episode in range(1, n_episodes):
        a = longest_queue_action(s, a)
        s, _, _, _ = env.step(a)


if __name__ == '__main__':
    main()
