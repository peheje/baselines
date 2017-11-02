import sys

sys.path.append('../')
import matplotlib.pyplot as plt
import gym
import numpy as np


def run_episode(env, parameters):
    observation = env.reset()
    total_reward = 0
    for _ in range(200):
        # Each weight is multiplied by its respective observation, and the products are summed up.
        # This is equivalent to performing an matrix multiplication of the two vectors.
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def enhance_observation(observation):
    observation_exp = np.empty(8)
    for i, _ in enumerate(observation):
        observation_exp[i] = observation[i]
    for i in range(4, 8):
        observation_exp[i] = observation[i - 4] ** 2
    return observation_exp


def train_random(env, n):
    """ Random search for best parameters """
    env.reset()
    counter = 0
    best_params = None
    best_reward = 0
    for _ in range(n):
        counter += 1
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env, parameters)
        if reward > best_reward:
            best_reward, best_params = reward, parameters
            if reward == 200:
                break
    return best_params, counter


def train_hillclimb(env, n):
    """ Hillclimb for best parameters """
    env.reset()
    start_noise = 0.1
    noise = 0.1
    parameters = np.random.rand(4) * 2 - 1
    best_reward = 0
    counter = 0
    for _ in range(n):
        counter += 1
        new_params = parameters + (np.random.rand(4) * 2 - 1) * noise
        reward = run_episode(env, new_params)
        if reward > best_reward:
            noise = start_noise
            best_reward, parameters = reward, new_params
            if reward == 200:
                break
        else:
            noise *= 20
    return parameters, counter


def create_graphs():
    """ Run n training and plot the number of episodes to reach reward of 200 and mean. """
    env = gym.make('CartPole-v0')
    n = 1000
    results = []
    for _ in range(n):
        results.append(train_hillclimb(env, n)[1])
    plt.hist(results, 50, normed=1, facecolor='g', alpha=0.75)
    plt.xlabel('Episodes required to reach 200')
    plt.ylabel('Frequency')
    plt.title('Histogram of Search')
    print("mean episodes to reach reward of 200: {}".format(np.sum(results) / n))
    plt.show()


def play_game():
    """ Train n steps and play the game with best parameters. """
    n = 10000
    env = gym.make('CartPole-v0')
    parameters, _ = train_random(env, n)
    observation = env.reset()
    step = 0
    while True:
        step += 1
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        if done:
            if step == 200:
                print("Completed")
            step = 0
            env.reset()
        env.render()


# create_graphs()
play_game()
