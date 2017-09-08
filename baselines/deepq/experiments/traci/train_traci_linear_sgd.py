# Inspired by https://bitbucket.org/chrisyeh96/cs-229-325-project
import gym
import TraciSimpleEnv.TraciSimpleEnv
import numpy as np
from sklearn.linear_model import SGDRegressor

env = gym.make('TraciSimpleEnv-v0')

print("made gym")


class LinearQFunction:
    def __init__(self):

        # Hyperparameters
        self.gamma = 0.99

        # Get a model for each action to take, based upon schema (feature vector)
        self.schema = ["right_cars_waiting", "left_cars_waiting", "top_cars_waiting", "bottom_cars_waiting"]
        self.n_actions = 3
        self.models = [SGDRegressor(alpha=1.0, eta0=0.0001) for i in range(self.n_actions)]
        base_x = [[1 for _ in self.schema]]
        base_y = []
        for model in self.models:
            model.fit(base_x, base_y)

    def get_q(self, features):
        features = np.array(features)
        q = [model.predict([features]) for model in self.models]
        return q

    def get_best_q(self, features):
        return np.max(self.get_q(features))

    def train(self, features, actions, rewards, new_features):
        examples = zip(features, actions, rewards, new_features)
        targets = [
            r + self.gamma * self.get_best_q(new_f)
            for f, a, r, new_f in examples
        ]
        examples = zip(features, actions, targets)
        for action in range(self.n_actions):
            # Extract the feature-target pair for the specific action
            x = [
                f
                for f, a, t in examples
                if a == action
            ]
            y = [
                t
                for f, a, t in examples
                if a == action
            ]
            if len(x) > 0:
                x = np.array(x).astype(np.float)
                y = np.array(y).astype(np.float)
                self.models[action].partial_fit(x, y)
