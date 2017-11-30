import os
import gym

from BaseTraciEnv import BaseTraciEnv
import Traci_2_cross_env.Traci_2_cross_env
import Traci_3_cross_env.Traci_3_cross_env
from baselines import logger, logger_utils
from pathlib import Path


def main():
    # Setup logging
    env_name = "Traci_3_cross_env-v0"
    logger.reset()
    log_dir = [os.path.join(str(Path.home()), "Desktop"), env_name + "enjoy_cycle"]
    logger_path = logger_utils.path_with_date(log_dir[0], log_dir[1])
    logger.configure(logger_path, ["tensorboard", "stdout"])

    # Make the environment and configure it for enjoying
    env = gym.make(env_name)
    env.configure_traci(perform_actions=False,
                        num_car_chances=1000,
                        start_car_probabilities=[0.25, 0.05],
                        end_car_probabilities=[1.0,0.1],
                        enjoy_car_probs=False,
                        reward_func=env.reward_total_waiting_vehicles,
                        action_func=BaseTraciEnv.set_light_phase_4_cross_green_dir,
                        state_contain_num_cars_in_queue_history=True,
                        state_contain_time_since_tl_change=True,
                        state_contain_tl_state_history=True,
                        num_episodes_from_start_car_probs_to_end_car_probs=100,
                        state_contain_avg_speed_between_detectors_history=False,
                        num_actions_pr_trafficlight=2)

    # env.render()
    _, done = env.reset(), False
    for episode in range(100):
        while not done:
            _, _, done, _ = env.step(-1)
        _, done = env.reset(), False

    env.log_travel_time_table()


if __name__ == '__main__':
    # with Profiler():
    main()
