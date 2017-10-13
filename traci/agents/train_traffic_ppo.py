#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import BaseTraciEnv
import Traci_1_cross_env.Traci_1_cross_env
import Traci_2_cross_env.Traci_2_cross_env
import Traci_3_cross_env.Traci_3_cross_env
from pathlib import Path
from baselines import logger, logger_utils
import os
import tensorflow as tf
from shutil import copyfile

def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=4).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)

    env.configure_traci(num_car_chances=1000,
                        start_car_probabilities=[1.0, 0.1],
                        end_car_probabilities=[1.0, 0.1],
                        reward_func=BaseTraciEnv.BaseTraciEnv.reward_average_speed,
                        state_contain_num_cars_in_queue_history=True,
                        state_contain_avg_speed_between_detectors_history=False,
                        state_contain_time_since_tl_change=True,
                        state_contain_tl_state_history=True,
                        num_actions_pr_trafficlight=2)
    # Initialize logger
    # Setup path of logging, name of environment
    log_dir = [os.path.join(str(Path.home()), "Desktop"), 'Traci_3_cross_env-v0-ppo']
    logger_path = logger_utils.path_with_date(log_dir[0], log_dir[1])

    logger.reset()
    logger.configure(logger_path, ["tensorboard", "stdout"])
    copyfile(__file__, logger_path + "/params.txt")

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    U.save_state(logger_path+"/saved_model")

    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Traci_3_cross_env-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    train(args.env, num_timesteps=1e6, seed=args.seed)


if __name__ == '__main__':
    main()
