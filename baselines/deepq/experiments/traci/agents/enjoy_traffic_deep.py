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
    env = gym.make('Traci_2_cross_env-v0')
    act = deepq.load("/home/peter/Desktop/2017-09-18_15-40-58/model-2017-09-18_16-24-20.pkl")
    env.configure_traci(num_car_chances=10000,
                        car_props=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        reward_func=env.reward_total_in_queue)
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
