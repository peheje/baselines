import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import gym

from baselines import deepq
import Traci_2_cross_env.Traci_2_cross_env

def main():
    env = gym.make('Traci_3_cross_env-v0')
    act = deepq.load("/home/nikolaj/Desktop/Traci_2_cross_env-v0/2017-09-20_12-10-41/model-2017-09-20_13-04-51.pkl")
    env.configure_traci(num_car_chances=10000,
                        car_props=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        reward_func=env.reward_squared_wait_sum)
    env.render()

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:

            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
