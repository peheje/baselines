import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import gym
from baselines.common import set_global_seeds, tf_util as U
from baselines import logger,logger_utils
from baselines.ppo1 import mlp_policy, pposgd_simple
from pathlib import Path
import Traci_2_cross_env.Traci_2_cross_env
import Traci_3_cross_env.Traci_3_cross_env
import tensorflow as tf


def main():
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    environment='Traci_3_cross_env-v0'
    path_to_model="/home/nikolaj/Desktop/Traci_3_cross_env-v0-ppo/2017-10-01_11-56-54/saved_model"
    env = gym.make(environment)
    env.configure_traci(num_car_chances=10000,
                        start_car_probabilities=[1, 0.05],
                        reward_func=env.reward_squared_wait_sum,
                        state_contain_num_cars_in_queue_history=True,
                        state_contain_avg_speed_between_detectors_history=False,
                        state_contain_time_since_tl_change=True,
                        state_contain_tl_state_history=True,
                        num_actions_pr_trafficlight=2)

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)
    pi = policy_fn('pi', env.observation_space, env.action_space)
    tf.train.Saver().restore(sess, path_to_model)

    # Setup path of logging, name of environment and save the current arguments (this script)
    log_dir = [os.path.join(str(Path.home()), "Desktop"), environment+"enjoy"]
    logger_path = logger_utils.path_with_date(log_dir[0], log_dir[1])
    # Initialize logger
    logger.reset()
    logger.configure(logger_path, ["tensorboard", "stdout"])
    logger.logtxt(path_to_model,"Model path")

    env.render()
    obs, done = env.reset(), False
    for i in range(10):
        episode_rew = 0
        while not done:

            obs, rew, done, _ = env.step(pi.act(True,obs)[0])
            episode_rew += rew
        print("Episode reward", episode_rew)
        obs, done = env.reset(), False # Done afterwards to ensure logging
    env.log_travel_time_table()


if __name__ == '__main__':
    main()
