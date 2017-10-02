import os, sys
from shutil import copyfile

from datetime import datetime
import inspect
import tensorflow as tf

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import gym

from baselines import deepq
import Traci_1_cross_env.Traci_1_cross_env
import Traci_2_cross_env.Traci_2_cross_env
import Traci_3_cross_env.Traci_3_cross_env
from baselines import logger, logger_utils
from BaseTraciEnv import BaseTraciEnv
from pathlib import Path


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def train_and_log(environment="Traci_3_cross_env-v0",
                  car_chances=1000,
                  reward_function=BaseTraciEnv.reward_halting_in_queue_3cross,
                  lr=1e-3,
                  max_timesteps=int(1e6),
                  buffer_size=50000,
                  exploration_fraction=0.5,
                  explore_final_eps=0.02,
                  train_freq=10,
                  batch_size=32,
                  checkpoint_freq=int(10000),
                  learning_starts=10000,
                  gamma=0.9,
                  target_network_update_freq=500,
                  car_probabilities=[0.25, 0.05], # [0.1,0.1,0.1,0.1,0.1,0.1,0.1], #For traci_3_cross: Bigroad_spawn_prob,Smallroad_spawn_prob
                  prioritized_replay=False,
                  prioritized_replay_alpha=0.6,
                  prioritized_replay_beta0=0.4,
                  prioritized_replay_beta_iters=None,
                  prioritized_replay_eps=1e-6,
                  num_cpu=4,
                  param_noise=False,
                  state_use_queue_length_history=True,
                  state_use_tl_state_history=True,
                  state_use_time_since_tl_change=True,
                  state_use_avg_speed_history=False,
                  hidden_layers=[8, 8, 8],
                  num_actions_pr_trafficlight=3):
    # Print call values
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)

    call_params_string_array = ['function name "%s"' % inspect.getframeinfo(frame)[2]]
    for i in args:
        call_params_string_array.append("    %s = %s" % (i, values[i]))

    # Setup path of logging, name of environment and save the current arguments (this script)
    log_dir = [os.path.join(str(Path.home()), "Desktop"), environment]
    logger_path = logger_utils.path_with_date(log_dir[0], log_dir[1])

    # Create environment and initialize
    env = gym.make(log_dir[1])
    env.configure_traci(num_car_chances=car_chances,
                        car_props=car_probabilities,
                        reward_func=reward_function,
                        state_contain_num_cars_in_queue_history=state_use_queue_length_history,
                        state_contain_avg_speed_between_detectors_history=state_use_avg_speed_history,
                        state_contain_time_since_tl_change=state_use_time_since_tl_change,
                        state_contain_tl_state_history=state_use_tl_state_history,
                        num_actions_pr_trafficlight=num_actions_pr_trafficlight)
    #env.render()

    # Initialize logger
    logger.reset()
    logger.configure(logger_path, ["tensorboard", "stdout"])
    logger.logtxt(call_params_string_array)

    copyfile(__file__, logger_path + "/params.txt")

    # Create the training model
    model = deepq.models.mlp(hidden_layers)
    act = deepq.learn(
        env=env,
        q_func=model,
        lr=lr,
        max_timesteps=max_timesteps,
        buffer_size=buffer_size,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=explore_final_eps,
        train_freq=train_freq,
        batch_size=batch_size,
        print_freq=1,
        checkpoint_freq=checkpoint_freq,
        learning_starts=learning_starts,
        gamma=gamma,
        target_network_update_freq=target_network_update_freq,
        prioritized_replay=prioritized_replay,
        prioritized_replay_alpha=prioritized_replay_alpha,
        prioritized_replay_beta0=prioritized_replay_beta0,
        prioritized_replay_beta_iters=prioritized_replay_beta_iters,
        prioritized_replay_eps=prioritized_replay_eps,
        num_cpu=num_cpu,
        param_noise=param_noise,
        callback=None,
        model_path=logger_path
    )
    save_path = logger_path + "/model-" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".pkl"
    print("Saving last model to {}".format(save_path))
    act.save(save_path)


def main():

    reward_functions = [BaseTraciEnv.reward_total_waiting_vehicles,
                        BaseTraciEnv.reward_total_in_queue_3cross,
                        BaseTraciEnv.reward_arrived_vehicles,
                        BaseTraciEnv.reward_average_speed,
                        BaseTraciEnv.reward_halting_in_queue_3cross]

    for rf in reward_functions:
        print("Now reward function is:", rf)
        g = tf.Graph()
        sess = tf.InteractiveSession(graph=g)
        with g.as_default():
            train_and_log(reward_function=rf)


if __name__ == '__main__':
    main()
