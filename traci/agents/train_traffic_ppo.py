#!/usr/bin/env python
import inspect

from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
from BaseTraciEnv import BaseTraciEnv
import Traci_1_cross_env.Traci_1_cross_env
import Traci_2_cross_env.Traci_2_cross_env
import Traci_3_cross_env.Traci_3_cross_env
from pathlib import Path
from baselines import logger, logger_utils
import os
import tensorflow as tf
from shutil import copyfile

def train_and_log(env_id,
                  seed,
                  checkpoint_freq=10000,
                  num_car_chances=1000,
                  action_function=BaseTraciEnv.set_light_phase_4_cross_green_dir,
                  reward_function=BaseTraciEnv.reward_total_waiting_vehicles,
                  max_timesteps=int(1e6),
                  start_car_probabilities=[1.0, 0.1],
                  end_car_probabilities=None,  # When set to None do not anneal
                  num_steps_from_start_car_probs_to_end_car_probs=1e5,
                  num_cpu=8,
                  state_use_queue_length=True,
                  state_use_tl_state=True,
                  state_use_time_since_tl_change=True,
                  state_use_avg_speed=False,
                  num_actions_pr_trafficlight=2,
                  num_history_states=2,
                  hid_size=64,
                  num_hid_layers=2,
                  timesteps_per_batch=2048,
                  clip_param=0.2, entcoeff=0.0,
                  optim_epochs=10,
                  optim_stepsize=3e-4,
                  optim_batchsize=64,
                  gamma=0.99,
                  lam=0.95,
                  schedule='linear'
                  ):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    # Obtain call params
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)

    call_params_string_array = ['function name "%s"' % inspect.getframeinfo(frame)[2]]
    for i in args:
        call_params_string_array.append("    %s = %s" % (i, values[i]))

    U.make_session(num_cpu=num_cpu).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)

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
                        num_history_states=num_history_states)
    # Initialize logger
    # Setup path of logging, name of environment

    log_dir = [os.path.join(str(Path.home()), "Desktop"), 'Traci_3_cross_env-v0-ppo']
    logger_path = logger_utils.path_with_date(log_dir[0], log_dir[1])

    logger.reset()
    logger.configure(logger_path, ["tensorboard", "stdout"])
    logger.logtxt(call_params_string_array)
    copyfile(__file__, logger_path + "/params.txt")

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=hid_size, num_hid_layers=num_hid_layers)
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn,
                        max_timesteps=max_timesteps,
                        timesteps_per_batch=timesteps_per_batch,
                        clip_param=clip_param, entcoeff=entcoeff,
                        optim_epochs=optim_epochs, optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
                        gamma=gamma, lam=lam, schedule=schedule, checkpoint_freq=checkpoint_freq,logger_path=logger_path
                        )
    U.save_state(logger_path+"/ckpt/saved_model")

    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Traci_3_cross_env-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    train_and_log(args.env, max_timesteps=1e6, seed=args.seed,checkpoint_freq=100,num_car_chances=100)


if __name__ == '__main__':
    main()
