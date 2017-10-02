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


def main():
    environment='Traci_3_cross_env-v0'
    path_to_model="/home/nikolaj/Desktop/model-2017-09-26_22-41-59.pkl"
    env = gym.make(environment)
    act = deepq.load(path_to_model)
    env.configure_traci(num_car_chances=1000,
                        car_props=[0.25,0.05],
                        reward_func=env.reward_squared_wait_sum,
                        state_contain_num_cars_in_queue_history=True,
                        state_contain_avg_speed_between_detectors_history=True,
                        state_contain_time_since_tl_change=True,
                        state_contain_tl_state_history=True,
                        num_actions_pr_trafficlight=3)

    # Setup path of logging, name of environment and save the current arguments (this script)
    log_dir = [os.path.join(str(Path.home()), "Desktop"), environment+"enjoy"]
    logger_path = logger_utils.path_with_date(log_dir[0], log_dir[1])
    # Initialize logger
    logger.reset()
    logger.configure(logger_path, ["tensorboard", "stdout"])
    logger.logtxt(path_to_model,"Model path")

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
    main()
