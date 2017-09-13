import gym
import numpy as np
from baselines import deepq
import Traci_2_cross_env.Traci_2_cross_env
from baselines import logger, logger_utils


n_episodes = 10000000


def main():
    print_timestep_freq = 100
    logger.reset()
    logger_path = logger_utils.path_with_date("/tmp/Traci_2_cross_env-v0", "Traci_2_cross_env-v0")
    logger.configure(logger_path, ["tensorboard", "stdout"])

    env = gym.make('Traci_2_cross_env-v0')
    env.render()
    s = env.reset()

    for episode in range(1, n_episodes):
        # noop
        s, r, _, _ = env.step(8)

        if episode % print_timestep_freq == 0:
            logger.record_tabular("steps_timestep", episode)
            logger.record_tabular("reward_timestep", r)
            #logger.record_tabular("mean 100 timestep reward", np.mean(mean_100timestep_reward))
            #logger.record_tabular("% time spent exploring_timestep", int(100 * exploration.value(t)))
            logger.dump_tabular()


if __name__ == '__main__':
    main()
