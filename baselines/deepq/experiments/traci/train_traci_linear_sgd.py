# Inspired by https://bitbucket.org/chrisyeh96/cs-229-325-project
import random

import gym
import TraciSimpleEnv.TraciSimpleEnv
import numpy as np
from collections import deque
from sklearn.linear_model import SGDRegressor


class LinearQFunction:
    def __init__(self, gamma, n_actions):

        # Hyperparameters
        self.gamma = gamma

        # Get a model for each action to take, based upon schema (feature vector)
        self.schema = ["right_cars_waiting", "left_cars_waiting", "top_cars_waiting", "bottom_cars_waiting",
                       "traffic_light_status"]
        self.n_actions = n_actions
        self.models = [SGDRegressor(loss="squared_loss", penalty="l2", alpha=0.0001,
                                    l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None,
                                    shuffle=True, verbose=0, epsilon=0.1,
                                    random_state=None, learning_rate="invscaling", eta0=0.001,
                                    power_t=0.25, warm_start=False, average=False, n_iter=None) for _ in
                       range(self.n_actions)]
        base_x = [[1 for _ in self.schema]]
        base_y = [0]
        for model in self.models:
            model.fit(base_x, base_y)

    def get_q(self, features):
        features = np.array(features)
        q = [model.predict([features]) for model in self.models]
        return q

    def get_best_q(self, features):
        q = np.array(self.get_q(features))
        #print("q", q)
        return int(np.random.choice(np.where(q == q.max())[0])), np.max(q)

    def train(self, features, actions, rewards, new_features):
        examples = list(zip(features, actions, rewards, new_features))
        targets = [
            r + self.gamma * self.get_best_q(new_f)[1]
            for f, a, r, new_f in examples
        ]
        examples = list(zip(features, actions, targets))
        for action in range(self.n_actions):
            x, y = [], []
            for f, a, t in examples:
                if a == action:
                    x.append(f)
                    y.append(t)
            if len(x) > 0:
                x = np.array(x).astype(np.float)
                y = np.array(y).astype(np.float)
                self.models[action].partial_fit(x, y)


env = gym.make('Traci_2_cross_env-v0')
print("made gym")

gamma = 0.99
n_actions = env.action_space.n
n_episodes = 1000000
epsilon = 0.4
epsilon_decay = 0.9999
print_every = 500
train_every = 100
mini_batch_size = 32
max_batch_size = mini_batch_size*100
batch = deque(maxlen=max_batch_size)

qf = LinearQFunction(gamma=gamma, n_actions=n_actions)


s = env.reset()
for episode in range(1, n_episodes):
    a = np.random.randint(0, n_actions) if np.random.rand() < epsilon else qf.get_best_q(s)[0]
    sn, r, done, info = env.step(a)
    batch.append((s, a, r, sn))
    if episode % train_every == 0:
        mini_batch = random.sample(batch, mini_batch_size)
        batch_unpacked = list(zip(*mini_batch))
        qf.train(*batch_unpacked)
    s = np.copy(sn)
    epsilon *= epsilon_decay

    if episode % print_every == 0:
        print("action taken {}".format(a))
        print("epsilon {}".format(epsilon))
        print("state {}".format(s))
        print("reward", r)
