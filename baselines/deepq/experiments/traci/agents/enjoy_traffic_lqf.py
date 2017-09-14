import gym
import numpy as np
from baselines import deepq
import Traci_1_cross_env.Traci_1_cross_env
from baselines import logger, logger_utils


n_episodes = 10000000


def longest_queue_action(state, old_action):
    print(state)
    cars_ns = state[1] + state[3]
    cars_we = state[0] + state[2]
    if cars_ns == cars_we:
        # print("same amount, keep light as is")
        return old_action
    elif cars_ns > cars_we:
        # print("most cars from north or south")
        return 0
    else:
        # print("most cars from west or east")
        return 1


def main():
    print_timestep_freq = 100
    logger.reset()
    logger_path = logger_utils.path_with_date("/tmp/Traci_1_cross_env-v0", "Traci_1_cross_env-v0")
    logger.configure(logger_path, ["tensorboard", "stdout"])

    env = gym.make('Traci_1_cross_env-v0')
    s = env.reset()
    a = 0

    for episode in range(1, n_episodes):
        a = longest_queue_action(s, a)
        s, r, _, _ = env.step(a)

        if episode % print_timestep_freq == 0:
            logger.record_tabular("steps_timestep", episode)
            logger.record_tabular("reward_timestep", r)
            #logger.record_tabular("mean 100 timestep reward", np.mean(mean_100timestep_reward))
            #logger.record_tabular("% time spent exploring_timestep", int(100 * exploration.value(t)))
            logger.dump_tabular()


if __name__ == '__main__':
    main()
