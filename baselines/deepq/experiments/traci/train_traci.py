import gym
import TraciSimpleEnv.TraciSimpleEnv
env = gym.make('TraciSimpleEnv-v0')

print("made gym")

while True:
    obs, rew, done, info = env.step(env.action_space.sample())

    print(obs, rew, done, info)