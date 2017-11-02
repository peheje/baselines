import gym
import numpy as np
import tensorflow as tf

env = gym.make('flashgames.Wheelers-v0')
env.reset()

while True:
    obs, rew, done, debug = env.step(env.action_space.sample())
    if done:
        env.reset()
    env.render()