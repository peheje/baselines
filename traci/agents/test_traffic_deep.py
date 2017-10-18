import os, sys

from BaseTraciEnv import BaseTraciEnv

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import gym

from baselines import deepq,logger,logger_utils
from pathlib import Path
import Traci_2_cross_env.Traci_2_cross_env
import Traci_3_cross_env.Traci_3_cross_env
import tensorflow as tf


def test(environment_name, path_to_model, configured_environment, act=None, log_dir="", render=False):
    print("RUNNING TEST")
    env = configured_environment
    if act is None:
        act = deepq.load(path_to_model)

    # Setup path of logging, name of environment and save the current arguments (this script)
    if log_dir == "":
        log_dir = os.path.join(str(Path.home()), "Desktop")
    logger_path = logger_utils.path_with_date(log_dir, environment_name+"test_deep_q")

    # Initialize logger
    logger.reset()
    logger.configure(logger_path, ["tensorboard", "stdout"])
    logger.logtxt(path_to_model,"model_path")

    # Run episodes acting greedily
    if render:
        env.render()

    obs, done = env.reset(), False
    for i in range(4):
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)
        obs, done = env.reset(), False # Done afterwards to ensure logging
    env.log_travel_time_table()


if __name__ == '__main__':
    # If run as main
    paths_to_model=["/home/peter/Desktop/Traci_3_cross_env-v0deep_q/2017-10-18_11-11-17/model-2017-10-18_13-15-49_reward_timesteps.pkl"]

    environment_name = 'Traci_3_cross_env-v0'
    env = gym.make(environment_name)
    env.configure_traci(num_car_chances=2000,
                        start_car_probabilities=[1.0, 0.1],
                        enjoy_car_probs=False,
                        reward_func=BaseTraciEnv.reward_total_in_queue_3cross,
                        state_contain_num_cars_in_queue_history=True,
                        state_contain_time_since_tl_change=False,
                        state_contain_tl_state_history=False,
                        state_contain_avg_speed_between_detectors_history=False,
                        num_actions_pr_trafficlight=2,
                        num_history_states=1)

    for path in paths_to_model:
        test(environment_name=environment_name,
             path_to_model=path,
             configured_environment=env,
             render=True)


