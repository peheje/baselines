import os, sys
from shutil import copyfile

from datetime import datetime

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import gym

from baselines import deepq
import Traci_1_cross_env.Traci_1_cross_env
import Traci_2_cross_env.Traci_2_cross_env
from baselines import logger, logger_utils
from pathlib import Path



def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    # Setup path of logging, name of environment and save the current arguments (this script)
    log_dir = [os.path.join(str(Path.home()), "Desktop"), "Traci_2_cross_env-v0"]
    logger_path = logger_utils.path_with_date(log_dir[0], log_dir[1])

    # Create environment and initialize
    env = gym.make(log_dir[1])
    env.configure_traci(num_car_chances=1000,
                        car_props=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        reward_func=env.reward_squared_wait_sum)
    # env.render()

    # Initialize logger
    logger.reset()
    logger.configure(logger_path, ["tensorboard", "stdout"])
    copyfile(__file__, logger_path + "/params.txt")

    # Create the training model
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env=env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.8,
        exploration_final_eps=0.02,
        train_freq=100,
        batch_size=32,
        print_freq=1,
        checkpoint_freq=5000,
        learning_starts=1000,
        gamma=0.9,
        target_network_update_freq=500,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        num_cpu=4,
        param_noise=False,
        callback=None,
        model_path=logger_path
    )
    save_path = logger_path + "/model-" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".pkl"
    print("Saving model to {}".format(save_path))
    act.save(log_dir[1])


if __name__ == '__main__':
    main()
