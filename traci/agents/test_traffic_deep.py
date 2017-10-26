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


def test(environment_name, path_to_model, configured_environment, act=None, log_dir="", no_explore=False):
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
    # env.render()
    obs, done = env.reset(), False
    for i in range(5):
        episode_rew = 0
        while not done:
            if no_explore:
                obs, rew, done, _ = env.step(act(obs[None], update_eps=0)[0])
            else:
                obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)
        obs, done = env.reset(), False # Done afterwards to ensure logging
    env.log_travel_time_table()


if __name__ == '__main__':
    # If run as main

    path_props = [
        {
            "path": "/home/phj-nh/Desktop/architectures_zero_expl_test/2017-10-23_15-44-50_[16]_low/model-2017-10-23_17-40-58.pkl",
            "info": "[16]_LOW",
            "props": [0.25, 0.05]
        },
        {
            "path": "/home/phj-nh/Desktop/architectures_zero_expl_test/2017-10-24_01-29-43_[16]_high/model-2017-10-24_05-07-15.pkl",
            "info": "[16]_HIGH",
            "props": [1.0, 0.10]
        },
        {
            "path": "/home/phj-nh/Desktop/architectures_zero_expl_test/2017-10-23_17-41-41_[64]_low/model-2017-10-23_19-29-50.pkl",
            "info": "[64]_LOW",
            "props": [0.25, 0.05]
        },
        {
            "path": "/home/phj-nh/Desktop/architectures_zero_expl_test/2017-10-24_05-11-10_[64]_high/model-2017-10-24_08-33-51.pkl",
            "info": "[64]_HIGH",
            "props": [1.0, 0.10]
        },
        {
            "path": "/home/phj-nh/Desktop/architectures_zero_expl_test/2017-10-23_19-30-38_[1024]_low/model-2017-10-23_21-22-29.pkl",
            "info": "[1024]_LOW",
            "props": [0.25, 0.05]
        },
        {
            "path": "/home/phj-nh/Desktop/architectures_zero_expl_test/2017-10-24_10-13-56_[1024]_high/model-2017-10-24_14-43-40.pkl",
            "info": "[1024]_HIGH",
            "props": [1.0, 0.10]
        },
        {
            "path": "/home/phj-nh/Desktop/architectures_zero_expl_test/2017-10-23_21-23-12_[512,512]_low/model-2017-10-23_23-13-41.pkl",
            "info": "[512, 512]_LOW",
            "props": [0.25, 0.05]
        },
        {
            "path": "/home/phj-nh/Desktop/architectures_zero_expl_test/2017-10-25_11-27-22_[512,512]_high/model-2017-10-25_15-43-35.pkl",
            "info": "[512, 512]_HIGH",
            "props": [1.0, 0.10]
        },
        {
            "path": "/home/phj-nh/Desktop/architectures_zero_expl_test/2017-10-23_23-14-37_[256,256,256]_low/model-2017-10-24_01-28-21.pkl",
            "info": "[256, 256, 256]_LOW",
            "props": [0.25, 0.05]
        },
        {
            "path": "/home/phj-nh/Desktop/architectures_zero_expl_test/2017-10-25_16-10-25_[256,256,256]_high/model-2017-10-25_20-32-03.pkl",
            "info": "[256, 256, 256]_HIGH",
            "props": [1.0, 0.10]
        }
    ]

    for setup in path_props:
        path = setup["path"]
        props = setup["props"]
        info = setup["info"]
        print("Running test for: ", info)
        print(props, path)

        environment_name = 'Traci_3_cross_env-v0'
        env = gym.make(environment_name)
        env.configure_traci(num_car_chances=1000,
                            start_car_probabilities=props,
                            enjoy_car_probs=False,
                            reward_func=BaseTraciEnv.reward_total_waiting_vehicles,
                            action_func=BaseTraciEnv.set_light_phase_4_cross_green_dir,
                            state_contain_num_cars_in_queue_history=True,
                            state_contain_time_since_tl_change=True,
                            state_contain_tl_state_history=True,
                            state_contain_avg_speed_between_detectors_history=False,
                            num_actions_pr_trafficlight=2,
                            num_history_states=2)

        with tf.device("/gpu:1"):
            g = tf.Graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.InteractiveSession(graph=g, config=config)
            with g.as_default():
                test(environment_name=environment_name,
                     path_to_model=path,
                     configured_environment=env)


