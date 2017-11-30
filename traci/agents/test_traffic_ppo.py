import os, sys

from BaseTraciEnv import BaseTraciEnv

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import gym
from baselines.common import set_global_seeds, tf_util as U
from baselines import logger, logger_utils
from baselines.ppo1 import mlp_policy, pposgd_simple
from pathlib import Path
import Traci_2_cross_env.Traci_2_cross_env
import Traci_3_cross_env.Traci_3_cross_env
import tensorflow as tf


def test(environment_name, path_to_model, configured_environment, act, log_dir, n_test_episodes=5):
    def policy_fn(name, ob_space, ac_space, tls_id, process_id):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2, tls_id=tls_id, process_id=process_id)  # set these?

    if act is None:
        act = [{} for _ in range(4)]
        process_id = path_to_model.split("_")[-1]
        for i in range(4):  # Assuming 4 networks
            g = tf.Graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.InteractiveSession(graph=g, config=config)
            with g.as_default():
                act[i]["sess"] = sess
                act[i]["pi"] = policy_fn('pi' + str(process_id), configured_environment.observation_space,
                                         configured_environment.action_space, tls_id=i, process_id=process_id)
                tf.train.Saver().restore(sess, path_to_model + "/tls" + str(i) + "/ckpt/saved_model")
    else:
        # sort the acts based on tls id
        act = [x for x in sorted(act, key=lambda k: k["tls_id"])]
    # Setup path of logging, name of environment and save the current arguments (this script)
    if log_dir == "":
        log_dir = os.path.join(str(Path.home()), "Desktop")
    logger_path = logger_utils.path_with_date(log_dir, environment_name + "test_ppo")
    # Initialize logger
    logger.reset()
    logger.configure(logger_path, ["tensorboard", "stdout"])
    logger.logtxt(path_to_model, "Model path")
    # configured_environment.render()
    obs, done = configured_environment.reset(), False
    for i in range(n_test_episodes):
        episode_rew = 0
        while not done:
            actions = [None for _ in range(len(act))]
            for i in range(len(act)):
                with act[i]['sess'].as_default():
                    actions[i] = act[i]['pi'].act(True, obs)[0]
            obs, rew, done, _ = configured_environment.step(actions)
            # episode_rew += rew
        print("Episode reward", episode_rew)
        obs, done = configured_environment.reset(), False  # Done afterwards to ensure logging
    configured_environment.log_travel_time_table()


if __name__ == '__main__':
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    environment = 'Traci_3_cross_env-v0'
    path_to_model = "/home/phj-nh/Desktop/Traci_3_cross_env-v0-ppo-multiple-generalize/2017-11-24_12-49-44_pid_0"
    env = gym.make(environment)
    env.configure_traci(num_car_chances=1000,
                        start_car_probabilities=[0.1, 0.01],
                        end_car_probabilities=[1.0, 0.1],
                        num_episodes_from_start_car_probs_to_end_car_probs=100,
                        enjoy_car_probs=False,
                        reward_func=BaseTraciEnv.reward_average_speed_split,
                        action_func=BaseTraciEnv.set_light_phase_4_cross_green_dir,
                        state_contain_num_cars_in_queue_history=True,
                        state_contain_time_since_tl_change=True,
                        state_contain_tl_state_history=True,
                        state_contain_avg_speed_between_detectors_history=False,
                        num_actions_pr_trafficlight=2,
                        num_history_states=1)
    test(environment, path_to_model, env, None, "", n_test_episodes=100)
