import sys
import os
from collections import deque

import gym

import traci
import numpy as np
from baselines import logger

# we need to import python modules from the $SUMO_HOME/tools directory
from utilities.UniqueCounter import UniqueCounter

try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")


class BaseTraciEnv(gym.Env):
    def __init__(self):
        self.episode_rewards = deque([0.0], maxlen=100)
        self.step_rewards = deque([], maxlen=100)
        self.co2_step_rewards = []
        self.avg_speed_step_rewards = []
        self.fully_stopped_cars = UniqueCounter()
        self.has_driven_cars = UniqueCounter()  # Necessary to figure out how many cars have stopped in simulation
        self.timestep = 0
        self.episode = 0
        self.traffic_light_changes = 0

        # configure_traci() must be called, leave uninitialized here
        self.num_car_chances = None
        self.car_props = None
        self.reward_func = None

    def configure_traci(self, num_car_chances, car_props, reward_func):
        self.num_car_chances = num_car_chances
        self.car_props = car_props
        self.reward_func = reward_func
        self.restart()

    def _reset(self):
        self.traffic_light_changes = 0
        # self.timestep = 0
        self.co2_step_rewards = []
        self.avg_speed_step_rewards = []
        self.fully_stopped_cars = UniqueCounter()  # Reset cars in counter
        self.has_driven_cars = UniqueCounter()  # Necessary to figure out how many cars have stopped in simulation

    def _step(self, action):
        """ Implement in child """
        raise NotImplementedError()

    def restart(self):
        """ Implement in child """
        raise NotImplementedError()

    @staticmethod
    def reward_total_waiting_vehicles():
        vehs = traci.vehicle.getIDList()
        total_wait_time = 0.0
        for veh_id in vehs:
            if traci.vehicle.getSpeed(veh_id) < 1:
                total_wait_time += 1.0
        return -total_wait_time

    @staticmethod
    def reward_emission():
        vehs = traci.vehicle.getIDList()
        emissions = []
        for veh_id in vehs:
            emissions.append(traci.vehicle.getCO2Emission(veh_id))
        if len(vehs) > 0:
            return -np.mean(emissions)
        else:
            return 0

    def reward_total_in_queue(self):
        num_waiting_cars = 0
        for counter in self.unique_counters:
            num_waiting_cars += counter.get_count()
        return -num_waiting_cars

    @staticmethod
    def reward_squared_wait_sum():
        vehs = traci.vehicle.getIDList()
        wait_sum = 0
        for veh_id in vehs:
            wait_sum += traci.vehicle.getWaitingTime(veh_id)
        return -np.mean(np.square(wait_sum))

    @staticmethod
    def reward_average_speed():
        vehs = traci.vehicle.getIDList()
        speed_sum = 0
        for veh_id in vehs:
            speed_sum += traci.vehicle.getSpeed(veh_id)
        if len(vehs) > 0:
            return speed_sum / len(vehs)
        else:
            return 0

    @staticmethod
    def discrete_to_multidiscrete(action, num_actions):
        mod_rest = action % num_actions
        div_floor = action // num_actions
        return [mod_rest, div_floor]

    def set_light_phase(self, light_id, action, cur_phase):
        # Run action
        if action == 0:
            if cur_phase == 2:
                traci.trafficlights.setPhase(light_id, 3)
                self.traffic_light_changes += 1
            elif cur_phase == 0:
                traci.trafficlights.setPhase(light_id, 0)
        elif action == 1:
            if cur_phase == 0:
                traci.trafficlights.setPhase(light_id, 1)
                self.traffic_light_changes += 1
            elif cur_phase == 2:
                traci.trafficlights.setPhase(light_id, 2)
        else:
            pass  # do nothing

    def add_fully_stopped_cars(self):
        vehs = traci.vehicle.getIDList()
        for veh_id in vehs:
            speed = traci.vehicle.getSpeed(veh_id)
            # this line can also be used for checking if car has breaklights on (8)stopped=traci.vehicle.getSignals(veh_id)
            if speed == 0 and self.has_driven_cars.contains(veh_id):
                self.fully_stopped_cars.add(veh_id)
            else:
                self.has_driven_cars.add(veh_id)

    def log_end_step(self, reward):
        # Calculate different rewards for step
        emission_reward = self.reward_emission()
        avg_speed_reward = self.reward_average_speed()
        self.add_fully_stopped_cars()

        self.episode_rewards[-1] += reward
        self.step_rewards.append(reward)
        self.co2_step_rewards.append(emission_reward)
        self.avg_speed_step_rewards.append(avg_speed_reward)
        mean_100step_reward = round(np.mean(self.step_rewards), 1)
        if False:  # not logging this atm
            logger.record_tabular("Step[Timestep]", self.timestep)
            logger.record_tabular("Reward[Timestep]", reward)
            logger.record_tabular("Mean 100 step reward[Timestep]", mean_100step_reward)
            logger.record_tabular("CO2-emission step reward[Timestep]", emission_reward)
            logger.record_tabular("Average-speed step reward[Timestep]", avg_speed_reward)
            # This cant be done here - logger.record_tabular("% time spent exploring[Timestep]", int(100 * exploration.value(t)))
            logger.dump_tabular()

        self.timestep += 1

    def log_end_episode(self, reward):

        mean_100ep_reward = round(np.mean(self.episode_rewards), 1)
        logger.record_tabular("Steps[Episode]", self.timestep)
        logger.record_tabular("Episodes[Episode]", self.episode)
        logger.record_tabular("Mean 100 episode reward[Episode]", mean_100ep_reward)
        logger.record_tabular("Episode Reward[Episode]", self.episode_rewards[-1])
        logger.record_tabular("Number of traffic light changes[Episode]", self.traffic_light_changes)
        logger.record_tabular("Total CO2 reward for episode[Episode]", sum(self.co2_step_rewards))
        logger.record_tabular("Total Avg-speed reward for episode[Episode]", sum(self.avg_speed_step_rewards))
        logger.record_tabular("Total number of stopped cars for episode[Episode]", self.fully_stopped_cars.get_count())
        # This cant be done here - logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
        logger.dump_tabular()
        self.episode_rewards.append(reward)
        self.episode += 1

    def do_logging(self, reward, done):
        self.log_end_step(reward)
        if done:
            self.log_end_episode(reward)
