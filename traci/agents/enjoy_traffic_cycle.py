import os
import gym
import numpy as np
from baselines import deepq
import Traci_2_cross_env.Traci_2_cross_env
import Traci_3_cross_env.Traci_3_cross_env
from baselines import logger, logger_utils
from pathlib import Path

from utilities.profiler import Profiler

n_episodes = 10000000


def main():
    env_name = "Traci_3_cross_env-v0"
    print_timestep_freq = 100
    logger.reset()

    log_dir = [os.path.join(str(Path.home()), "Desktop"), env_name]
    logger_path = logger_utils.path_with_date(log_dir[0], log_dir[1])
    logger.configure(logger_path, ["tensorboard", "stdout"])

    env = gym.make(env_name)

    env.configure_traci(num_car_chances=1000,
                        car_props=[0.25, 0.05],
                        reward_func=env.reward_total_waiting_vehicles,
                        num_actions_pr_trafficlight=2,
                        perform_actions=False)

    env.render()
    s = env.reset()

    for episode in range(1, n_episodes):
        # noop
        s, r, done, _ = env.step(-1)
        #if episode % print_timestep_freq == 0:
        #    logger.record_tabular("steps_timestep", episode)
        #    logger.record_tabular("reward_timestep", r)
        #    # logger.record_tabular("mean 100 timestep reward", np.mean(mean_100timestep_reward))
        #    # logger.record_tabular("% time spent exploring_timestep", int(100 * exploration.value(t)))
        #    logger.dump_tabular()
        if done:
            env.reset()


if __name__ == '__main__':
    #with Profiler():
    main()
