import gym
import TraciSimpleEnv.TraciSimpleEnv
env = gym.make('TraciSimpleEnv-v0')

print("made gym")

while True:
    obs, rew, done, info = env.step(0)

    print(obs, rew, done, info)

    if done:
        env.reset()