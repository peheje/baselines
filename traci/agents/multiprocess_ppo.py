import argparse
from multiprocessing import Process
from BaseTraciEnv import BaseTraciEnv
from agents.train_traffic_ppo import setup_thread_and_run

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='Traci_3_cross_env-v0')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
args = parser.parse_args()

reward_functions = [BaseTraciEnv.reward_average_speed,
                    BaseTraciEnv.reward_average_accumulated_wait_time,
                    BaseTraciEnv.reward_rms_accumulated_wait_time,
                    BaseTraciEnv.reward_total_waiting_vehicles,
                    BaseTraciEnv.reward_total_in_queue_3cross,
                    BaseTraciEnv.reward_arrived_vehicles,
                    BaseTraciEnv.reward_halting_in_queue_3cross]

probabilities = [[0.25, 0.05], [1.0, 0.10]]
process_list=[]
num_spawned_process=0

for rf in reward_functions:
    for pr in probabilities:
        my_t = Process(target=setup_thread_and_run,
                       kwargs={"reward_function": rf, "start_car_probabilities": pr,
                               "env_id": args.env,
                               "seed": args.seed,
                               "process_id":num_spawned_process})
        my_t.start()
        num_spawned_process+=1
        if num_spawned_process>=3:
            pass
            #my_t.join()
        process_list.append(my_t)
for i in range(len(process_list)):
    process_list[i].join()