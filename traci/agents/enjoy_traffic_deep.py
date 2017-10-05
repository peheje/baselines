import os, sys
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


def main(path_to_model):
    environment='Traci_3_cross_env-v0'
    env = gym.make(environment)
    act = deepq.load(path_to_model)
    env.configure_traci(num_car_chances=1000,
                        start_car_probabilities=[0.25, 0.05],
                        enjoy_car_probs=False,
                        reward_func=env.reward_total_waiting_vehicles,
                        state_contain_num_cars_in_queue_history=True,
                        state_contain_time_since_tl_change=True,
                        state_contain_tl_state_history=True,
                        state_contain_avg_speed_between_detectors_history=False,
                        num_actions_pr_trafficlight=3)

    # Setup path of logging, name of environment and save the current arguments (this script)
    log_dir = [os.path.join(str(Path.home()), "Desktop"), environment+"enjoy"]
    logger_path = logger_utils.path_with_date(log_dir[0], log_dir[1])
    # Initialize logger
    logger.reset()
    logger.configure(logger_path, ["tensorboard", "stdout"])
    logger.logtxt(path_to_model,"model_path")

    #env.render()
    obs, done = env.reset(), False
    for i in range(10):
        episode_rew = 0
        while not done:

            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)
        obs, done = env.reset(), False # Done afterwards to ensure logging
    env.log_travel_time_table()


if __name__ == '__main__':
    paths_to_model=["/Users/phj/Desktop/Traci_3_cross_env-v0/2017-10-05_13-41-06/model-2017-10-05_13-41-26.pkl"]
    for path in paths_to_model:
        g = tf.Graph()
        sess = tf.InteractiveSession(graph=g)
        with g.as_default():
            main(path_to_model=path)


