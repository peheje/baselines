import json
import os, sys
from shutil import copyfile

from datetime import datetime
import inspect
import tensorflow as tf

from agents import test_traffic_deep

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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train_and_log(environment_name="Traci_3_cross_env-v0",
                  num_car_chances=1000,
                  action_function=BaseTraciEnv.set_light_phase_4_cross_green_dir,
                  reward_function=BaseTraciEnv.reward_total_waiting_vehicles,
                  lr=(1e-3 / 4.0),  # lr / 4.0 as described in (Tom Schaul 2016) when using prio. exp. repl.
                  max_timesteps=int(1e6),
                  buffer_size=200000,
                  exploration_initial_p=0.2,
                  exploration_fraction=0.5,
                  explore_final_eps=0.02,
                  train_freq=10,
                  batch_size=32,
                  checkpoint_freq=10000,
                  learning_starts=10000,
                  gamma=0.9,
                  target_network_update_freq=500,
                  start_car_probabilities=[1.0, 0.1],
                  # [0.1,0.1,0.1,0.1,0.1,0.1,0.1], #For traci_3_cross: Bigroad_spawn_prob,Smallroad_spawn_prob
                  end_car_probabilities=None,  # When set to None do not anneal
                  num_steps_from_start_car_probs_to_end_car_probs=1e5,
                  prioritized_replay=True,
                  prioritized_replay_alpha=0.6,
                  prioritized_replay_beta0=0.4,
                  prioritized_replay_beta_iters=None,
                  prioritized_replay_eps=1e-6,
                  num_cpu=8,
                  param_noise=False,
                  state_use_queue_length=True,
                  normalize_queue_lengths=False,
                  state_use_tl_state=True,
                  state_use_time_since_tl_change=True,
                  state_use_avg_speed=False,
                  hidden_layers=[64],
                  num_actions_pr_trafficlight=2,
                  num_history_states=1,
                  experiment_name="",
                  layer_norm=False):
    print("RUNNING train_and_log")

    # Print call values
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)

    call_params_string_array = ['function name "%s"' % inspect.getframeinfo(frame)[2]]
    for i in args:
        call_params_string_array.append("    %s = %s" % (i, values[i]))

    # Setup path of logging, name of environment and save the current arguments (this script)
    log_dir = [os.path.join(str(Path.home()), "Desktop"), environment_name + "deep_q_" + experiment_name]
    logger_path = logger_utils.path_with_date(log_dir[0], log_dir[1])

    # Create environment and initialize
    env = gym.make(environment_name)
    env.configure_traci(num_car_chances=num_car_chances,
                        start_car_probabilities=start_car_probabilities,
                        end_car_probabilities=end_car_probabilities,
                        num_steps_from_start_car_probs_to_end_car_probs=num_steps_from_start_car_probs_to_end_car_probs,
                        reward_func=reward_function,
                        action_func=action_function,
                        state_contain_num_cars_in_queue_history=state_use_queue_length,
                        state_contain_avg_speed_between_detectors_history=state_use_avg_speed,
                        state_contain_time_since_tl_change=state_use_time_since_tl_change,
                        state_contain_tl_state_history=state_use_tl_state,
                        num_actions_pr_trafficlight=num_actions_pr_trafficlight,
                        num_history_states=num_history_states,
                        normalize_queue_lengths=normalize_queue_lengths)
    # env.render()

    # Initialize logger
    logger.reset()
    logger.configure(logger_path, ["tensorboard", "stdout"])
    logger.logtxt(call_params_string_array)

    with open(logger_path + "/params.txt", 'w') as file:
        file.write(json.dumps(call_params_string_array, indent=4))
    copyfile(__file__, logger_path + "/script.txt")

    # Create the training model
    model = deepq.models.mlp(hidden_layers, layer_norm=layer_norm)
    act = deepq.learn(
        env=env,
        q_func=model,
        lr=lr,
        max_timesteps=max_timesteps,
        buffer_size=buffer_size,
        exploration_initial_p=exploration_initial_p,
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

    # Run test
    test_environment = gym.make(environment_name)
    test_environment.configure_traci(num_car_chances=num_car_chances,
                                     start_car_probabilities=start_car_probabilities,
                                     enjoy_car_probs=False,
                                     reward_func=reward_function,
                                     action_func=action_function,
                                     state_contain_num_cars_in_queue_history=state_use_queue_length,
                                     state_contain_time_since_tl_change=state_use_time_since_tl_change,
                                     state_contain_tl_state_history=state_use_tl_state,
                                     state_contain_avg_speed_between_detectors_history=state_use_avg_speed,
                                     num_actions_pr_trafficlight=num_actions_pr_trafficlight,
                                     num_history_states=num_history_states,
                                     normalize_queue_lengths=normalize_queue_lengths)
    test_traffic_deep.test(environment_name=environment_name,
                           path_to_model=save_path,
                           configured_environment=test_environment,
                           act=act,
                           log_dir=logger_path)


def main():
    probabilities = [
        [0.25, 0.05],
        [1.0, 0.10]
    ]

    # Set this by hand!
    experiment_name = ""

    for pr in probabilities:
        print("Now props:", pr)
        g = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(graph=g, config=config)
        with g.as_default():
            train_and_log(start_car_probabilities=pr,
                          experiment_name=experiment_name)


if __name__ == '__main__':
    main()
