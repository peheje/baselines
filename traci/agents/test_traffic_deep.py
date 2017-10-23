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


def test(environment_name, path_to_model, configured_environment, act=None, log_dir=""):
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
    #env.render()
    obs, done = env.reset(), False
    for i in range(5):
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None], update_eps=0)[0])
            episode_rew += rew
        print("Episode reward", episode_rew)
        obs, done = env.reset(), False # Done afterwards to ensure logging
    env.log_travel_time_table()


if __name__ == '__main__':
    # If run as main

    path_props = [
        {
            "path": "/home/phj-nh/Desktop/Traci_3_cross_env-v0deep_q/2017-10-21_20-40-50/model-2017-10-21_22-24-14.pkl",
            "info": "LOW_EXTEND",
            "props": [0.25, 0.05],
            "action_func": BaseTraciEnv.set_light_phase_4_cross_extend
        },
        {
            "path": "/home/phj-nh/Desktop/Traci_3_cross_env-v0deep_q/2017-10-21_22-25-36/model-2017-10-22_00-03-55.pkl",
            "info": "LOW_DIR",
            "props": [0.25, 0.05],
            "action_func": BaseTraciEnv.set_light_phase_4_cross_green_dir
        },
        {
            "path": "/home/phj-nh/Desktop/Traci_3_cross_env-v0deep_q/2017-10-22_00-05-19/model-2017-10-22_03-30-04.pkl",
            "info": "HIGH_EXTEND",
            "props": [1.0, 0.10],
            "action_func": BaseTraciEnv.set_light_phase_4_cross_extend
        },
        {
            "path": "/home/phj-nh/Desktop/Traci_3_cross_env-v0deep_q/2017-10-22_03-39-08/model-2017-10-22_07-24-23.pkl",
            "info": "HIGH_DIR",
            "props": [1.0, 0.10],
            "action_func": BaseTraciEnv.set_light_phase_4_cross_green_dir
        }
    ]

    for setup in path_props:
        path = setup["path"]
        props = setup["props"]
        action_func = setup["action_func"]
        info = setup["info"]
        print("Running test for: ", info)
        print(path, props, action_func)

        environment_name = 'Traci_3_cross_env-v0'
        env = gym.make(environment_name)
        env.configure_traci(num_car_chances=1000,
                            start_car_probabilities=props,
                            enjoy_car_probs=False,
                            reward_func=BaseTraciEnv.reward_total_waiting_vehicles,
                            action_func=action_func,
                            state_contain_num_cars_in_queue_history=True,
                            state_contain_time_since_tl_change=True,
                            state_contain_tl_state_history=True,
                            state_contain_avg_speed_between_detectors_history=False,
                            num_actions_pr_trafficlight=2,
                            num_history_states=2)

        g = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(graph=g, config=config)
        with g.as_default():
            test(environment_name=environment_name,
                 path_to_model=path,
                 configured_environment=env)


