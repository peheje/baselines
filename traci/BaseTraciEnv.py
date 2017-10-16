import json
import sys
import os
import tempfile
from collections import deque, OrderedDict
import xml.etree.ElementTree
import math
import copy

import gym

import numpy as np
from baselines import logger
import time

# we need to import python modules from the $SUMO_HOME/tools directory
from utilities.UniqueCounter import UniqueCounter
from baselines.common.schedules import LinearSchedule
from utilities.profiler import Profiler

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci


class BaseTraciEnv(gym.Env):
    def __init__(self):
        self.episode_rewards = deque([0.0], maxlen=100)
        self.step_rewards = deque([], maxlen=100)
        self.mean_episode_rewards = deque([], maxlen=100)
        self.episode_mean_travel_times = []
        self.episode_mean_time_losses = []
        self.co2_step_rewards = []
        self.avg_speed_step_rewards = []
        self.total_time_loss = 0
        self.total_travel_time = 0
        self.fully_stopped_cars_map = {}
        self.has_driven_cars_map = {}
        self.timestep = 0
        self.timestep_this_episode = 0
        self.episode = 0
        self.traffic_light_changes = 0
        self.time_since_tl_change = OrderedDict()

        # make temp file for tripinfo
        self.tripinfo_file_name = tempfile.NamedTemporaryFile(mode="w", delete=False).name

        # Must be configured in specific constructor
        self.num_trafficlights = None

        # configure_traci() must be called, leave uninitialized here
        self.num_car_chances = None
        self.car_probabilities = None
        self.reward_func = None
        self.num_actions_pr_trafficlight = None
        self.num_actions = None
        self.action_converter = None
        self.perform_actions = None
        self.state_use_time_since_tl_change = None
        self.state_use_avg_speed_between_detectors_history = None
        self.state_use_num_cars_in_queue_history = None
        self.e3ids = None

    def configure_traci(self,
                        num_car_chances,
                        start_car_probabilities,
                        reward_func,
                        num_actions_pr_trafficlight,
                        num_steps_from_start_car_probs_to_end_car_probs=1e5,
                        num_history_states=2,
                        end_car_probabilities=None,
                        enjoy_car_probs=False,
                        perform_actions=True,
                        state_contain_avg_speed_between_detectors_history=False,
                        state_contain_time_since_tl_change=False,
                        state_contain_tl_state_history=True,
                        state_contain_num_cars_in_queue_history=True):

        """

        :param num_car_chances: How many chances there are to spawning cars.
        :param start_car_probabilities: Start car probabilities to start annealing down to end_car_probabilities.
        :param reward_func: Which reward function to use. Only used for training.
        :param num_actions_pr_trafficlight: 2 (Switch, Nothing) or 3 (Green-NS, Green-WE, Nothing). Only used for training.
        :param num_steps_from_start_car_probs_to_end_car_probs: Number of steps to anneal from start_car_probabilities to end_car_probabilities.
        :param num_history_states: How many history states to include in the state. Only used for training.
        :param end_car_probabilities: The end probability for spawning cars when annealed num_steps_from_start_car_probs_to_end_car_probs steps. If set to None, do not anneal.
        :param enjoy_car_probs: Whether to change car probabilities to something hardcoded for enjoy (test) over the episode.
        :param perform_actions: Whether to perform actions based upon some model, for cycle set to False.
        :param state_contain_avg_speed_between_detectors_history:
        :param state_contain_time_since_tl_change:
        :param state_contain_tl_state_history:
        :param state_contain_num_cars_in_queue_history:
        """
        self.num_actions_pr_trafficlight = num_actions_pr_trafficlight
        self.num_actions = self.num_actions_pr_trafficlight ** self.num_trafficlights
        if self.num_actions_pr_trafficlight == 2:
            self.action_converter = self.binary_action
        elif self.num_actions_pr_trafficlight == 3:
            self.action_converter = self.ternary_action
        else:
            raise Exception("Not supported other than 2 or 3 actions pr. traffic light.")

        self.enjoy_car_probs=enjoy_car_probs
        self.num_car_chances = num_car_chances
        self.car_probabilities = copy.deepcopy(start_car_probabilities)
        self.start_car_probabilities = copy.deepcopy(start_car_probabilities)
        self.end_car_probabilities = end_car_probabilities
        self.num_steps_from_start_car_probs_to_end_car_probs=num_steps_from_start_car_probs_to_end_car_probs
        self.reward_func = reward_func
        self.perform_actions = perform_actions
        self.state_use_time_since_tl_change = state_contain_time_since_tl_change
        self.state_use_avg_speed_between_detectors_history = state_contain_avg_speed_between_detectors_history
        self.state_use_tl_state_history = state_contain_tl_state_history
        self.state_use_num_cars_in_queue_history = state_contain_num_cars_in_queue_history
        self.num_history_states = num_history_states

        self.restart()

    def _reset(self):
        self.traffic_light_changes = 0
        self.total_travel_time = 0
        self.total_time_loss = 0
        # self.timestep = 0
        self.co2_step_rewards = []
        self.avg_speed_step_rewards = []
        self.fully_stopped_cars_map = {}
        self.has_driven_cars_map = {}
        self.timestep_this_episode = 0
        self.set_new_car_probabilities()

    def set_new_car_probabilities(self):
        if self.end_car_probabilities is None or self.enjoy_car_probs:
            return
        else:
            for i in range(len(self.car_probabilities)):
                new_prop = LinearSchedule(self.num_steps_from_start_car_probs_to_end_car_probs,
                                       initial_p=self.start_car_probabilities[i],
                                       final_p=self.end_car_probabilities[i]).value(self.timestep)
                # print("new_prop", new_prop)
                self.car_probabilities[i]=new_prop

    def _step(self, action):
        """ Implement in child """
        raise NotImplementedError()

    def restart(self):
        """ Implement in child """
        raise NotImplementedError()

    def get_state_multientryexit_old(self):
        # Handle historical state data
        if self.e3ids is None:
            self.e3ids = traci.multientryexit.getIDList()
        cur_state = []

        if self.state_use_num_cars_in_queue_history:
            for id in self.e3ids:
                cur_state.append(len(traci.multientryexit.getLastStepVehicleIDs(id)))
        if self.state_use_avg_speed_between_detectors_history:
            for id in self.e3ids:
                cur_state.append(traci.multientryexit.getLastStepMeanSpeed(id))
        if self.state_use_tl_state_history:
            cur_state = cur_state + self.get_traffic_states()
        self.old_state.append(np.array(cur_state, dtype=float))

        ### Add non historical state data
        state_to_return = np.hstack(self.old_state)
        if self.state_use_time_since_tl_change:
            state_to_return = np.concatenate([state_to_return, list(self.time_since_tl_change.values())])

        return state_to_return

    def calculate_num_history_state_scalars(self):
        num = 0
        num_traffic_lights = len(traci.trafficlights.getIDList())
        if self.state_use_tl_state_history:
            num += num_traffic_lights
        if self.state_use_num_cars_in_queue_history:
            num += num_traffic_lights * 4
        if self.state_use_avg_speed_between_detectors_history:
            num += num_traffic_lights * 4
        return num

    def calculate_num_nonhistory_state_scalars(self):
        num = 0
        num_traffic_lights = len(traci.trafficlights.getIDList())
        if self.state_use_time_since_tl_change:
            num += num_traffic_lights
        return num

    @staticmethod
    def reward_total_waiting_vehicles():
        vehs = traci.vehicle.getIDList()
        total_wait_time = 0.0
        for veh_id in vehs:
            if traci.vehicle.getSpeed(veh_id) < 1:
                total_wait_time += 1.0
        return -total_wait_time

    def setup_subscriptions_for_departed(self):
        raw_sim = traci.simulation.getSubscriptionResults()
        departed_ids = raw_sim[traci.constants.VAR_DEPARTED_VEHICLES_IDS]
        for veh_id in departed_ids:
            traci.vehicle.subscribe(veh_id, self.vehicle_subs)

    @staticmethod
    def reward_emission():
        # Subscription edition
        raw_vehicle = traci.vehicle.getSubscriptionResults()
        co2s = BaseTraciEnv.extract_list(raw_vehicle, traci.constants.VAR_CO2EMISSION)
        if len(co2s) > 0:
            a_mean = -np.mean(co2s)
        else:
            a_mean = 0.0

        return a_mean

    @staticmethod
    def reward_total_in_queue_3cross():
        s = 0
        for id in traci.multientryexit.getIDList():
            s += len(traci.multientryexit.getLastStepVehicleIDs(id))
        return -s

    @staticmethod
    def reward_halting_in_queue_3cross():
        s = 0
        for id in traci.multientryexit.getIDList():
            s += traci.multientryexit.getLastStepHaltingNumber(id)
        return -s

    def reward_total_in_queue(self):
        num_waiting_cars = 0
        for counter in self.unique_counters:
            num_waiting_cars += counter.get_count()
        return -num_waiting_cars

    @staticmethod
    def reward_halting_sum():
        sum = 0
        for tl in traci.trafficlights.getIDList():
            for link in traci.trafficlights.getControlledLinks(tl):
                for lane in link[0]:
                    sum += traci.lane.getLastStepHaltingNumber(lane)

        return -sum

    @staticmethod
    def reward_arrived_vehicles():
        return traci.simulation.getArrivedNumber()

    @staticmethod
    def reward_squared_wait_sum():
        vehs = traci.vehicle.getIDList()
        wait_sum = 0
        for veh_id in vehs:
            wait_sum += np.square(traci.vehicle.getWaitingTime(veh_id))
        return -np.mean(wait_sum)

    @staticmethod
    def reward_average_speed():

        for veh_id in traci.simulation.getDepartedIDList():
            traci.vehicle.subscribe(veh_id, [traci.constants.VAR_SPEED])
        speed_map = traci.vehicle.getSubscriptionResults()
        speeds = BaseTraciEnv.extract_list(speed_map, traci.constants.VAR_SPEED)

        if len(speeds) < 1:
            speeds.append(0.0)

        a_mean = np.mean(speeds)
        return a_mean

    @staticmethod
    def reward_average_accumulated_wait_time():
        vehs = traci.vehicle.getIDList()
        accu_sum = 0
        for veh_id in vehs:
            accu_sum += traci.vehicle.getAccumulatedWaitingTime(veh_id)
        if len(vehs) > 0:
            return -(accu_sum / len(vehs))
        else:
            return 0
    @staticmethod
    def reward_rms_accumulated_wait_time():
        vehs = traci.vehicle.getIDList()
        square_sum = 0
        for veh_id in vehs:
            square_sum += traci.vehicle.getAccumulatedWaitingTime(veh_id)**2
        if len(vehs) > 0:
            return -math.sqrt(square_sum / len(vehs))
        else:
            return 0
    @staticmethod
    def reward_estimated_travel_time():
        lanes = traci.lane.getIDList()
        travel_reward_sum = 0
        for lane_id in lanes:
            travel_reward_sum = +traci.lane.getTraveltime(lane_id)
        return -travel_reward_sum

    @staticmethod
    def discrete_to_multidiscrete_2cross(action, num_actions):
        mod_rest = action % num_actions
        div_floor = action // num_actions
        return [mod_rest, div_floor]

    # Assuming only 2 viable actions
    @staticmethod
    def binary_action(action):
        return list(map(int, format(action, '04b')))

    @staticmethod
    def extract_list(m, subscription_id):
        """ Extracts list from traci subscription result m """
        values = [m[id][subscription_id] for id in sorted(m)]
        return values

    @staticmethod
    def ternary_action(n):
        """ https://stackoverflow.com/questions/34559663/convert-decimal-to-ternarybase3-in-python """
        if n == 0:
            return [0, 0, 0, 0]
        nums = []
        while n:
            n, r = divmod(n, 3)
            nums.append(r)
        for i in range(4 - len(nums)):
            nums.append(0)
        return list(reversed(nums))

    def set_light_phase_4_cross(self, light_id, action, cur_phase):
        if light_id in self.time_since_tl_change:
            self.time_since_tl_change[light_id] += 1
        else:
            self.time_since_tl_change[light_id] = 1
        # Run action
        if action == 0:
            if cur_phase == 0:
                traci.trafficlights.setPhase(light_id, 0)
            elif cur_phase == 4:
                traci.trafficlights.setPhase(light_id, 4)
        elif action == 1:
            if cur_phase == 4:
                traci.trafficlights.setPhase(light_id, 5)
                self.traffic_light_changes += 1
                self.time_since_tl_change[light_id] = 0
            if cur_phase == 0:
                traci.trafficlights.setPhase(light_id, 1)
                self.traffic_light_changes += 1
                self.time_since_tl_change[light_id] = 0
        else:
            pass  # do nothing

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
        # Subscriptions and map
        raw_vehicle = traci.vehicle.getSubscriptionResults()
        speeds = self.extract_list(raw_vehicle, traci.constants.VAR_SPEED)

        veh_ids = traci.vehicle.getIDList()
        for s, veh_id in zip(speeds, veh_ids):
            if s == 0.0 and veh_id in self.has_driven_cars_map:
                self.fully_stopped_cars_map[veh_id] = True
            elif s > 0.0:
                self.has_driven_cars_map[veh_id] = True

    def log_end_step(self, reward):
        # print("Logging end step: ",self.timestep)
        # Calculate different rewards for step
        emission_reward = self.reward_emission()
        avg_speed_reward = self.reward_average_speed()
        self.add_fully_stopped_cars()

        self.episode_rewards[-1] += reward
        self.step_rewards.append(reward)
        self.co2_step_rewards.append(emission_reward)
        self.avg_speed_step_rewards.append(avg_speed_reward)
        mean_100step_reward = round(np.mean(self.step_rewards), 1)

        self.timestep_this_episode += 1
        self.timestep += 1

        if False:  # not logging this atm
            logger.record_tabular("Step[Timestep]", self.timestep)
            logger.record_tabular("Reward[Timestep]", reward)
            logger.record_tabular("Mean 100 step reward[Timestep]", mean_100step_reward)
            logger.record_tabular("CO2-emission step reward[Timestep]", emission_reward)
            logger.record_tabular("Average-speed step reward[Timestep]", avg_speed_reward)
            # This cant be done here - logger.record_tabular("% time spent exploring[Timestep]", int(100 * exploration.value(t)))
            logger.dump_tabular()

    def log_end_episode(self, reward):

        # Read tripinfo file
        retry = True
        while retry:
            try:
                e = xml.etree.ElementTree.parse(self.tripinfo_file_name).getroot()
                tripinfos = e.findall('tripinfo')
                num_tripinfos = len(tripinfos)
                for tripinfo in tripinfos:
                    self.total_travel_time += float(tripinfo.get('duration'))
                    self.total_time_loss += float(tripinfo.get('timeLoss'))
                retry = False
                logger.record_tabular("Total travel time for episode[Episode]", self.total_travel_time)
                logger.record_tabular("Total time loss for episode[Episode]", self.total_time_loss)
                self.episode_mean_travel_times.append(self.total_travel_time / num_tripinfos)
                self.episode_mean_time_losses.append(self.total_time_loss / num_tripinfos)
            except xml.etree.ElementTree.ParseError as err:
                print("Couldn't read xml, skipping logging this iteration")
                retry = False

        self.mean_episode_rewards.append(self.episode_rewards[-1] / self.timestep_this_episode)
        mean_100ep_mean_reward = round(np.mean(self.mean_episode_rewards), 1)

        mean_100ep_reward = round(np.mean(self.episode_rewards), 1)
        logger.record_tabular("Steps[Episode]", self.timestep)
        logger.record_tabular("Episodes[Episode]", self.episode)
        logger.record_tabular("Mean 100 episode reward[Episode]", mean_100ep_reward)
        logger.record_tabular("Mean 100 episode mean reward[Episode]", mean_100ep_mean_reward)
        logger.record_tabular("Episode Reward[Episode]", self.episode_rewards[-1])
        logger.record_tabular("Episode Reward mean[Episode]", self.mean_episode_rewards[-1])
        logger.record_tabular("Number of traffic light changes[Episode]", self.traffic_light_changes)
        logger.record_tabular("Total CO2 reward for episode[Episode]", sum(self.co2_step_rewards))
        logger.record_tabular("Mean Avg-speed reward for episode[Episode]", np.mean(self.avg_speed_step_rewards))

        fully_stopped = len(self.fully_stopped_cars_map)
        logger.record_tabular("Total number of stopped cars for episode[Episode]", fully_stopped)
        logger.dump_tabular()
        self.episode_rewards.append(reward)
        self.episode += 1

    def do_logging(self, reward, done):
        self.log_end_step(reward)
        if done:
            self.log_end_episode(reward)

    def log_travel_time_table(self):
        string_array_to_log = [["Episode", "Mean travel time", "Mean time loss"]]
        for i in range(len(self.episode_mean_travel_times)):
            string_array_to_log.append(
                [str(i + 1), str(self.episode_mean_travel_times[i]), str(self.episode_mean_time_losses[i])])

        # Means
        string_array_to_log.append(
            ["Means", str(np.mean(self.episode_mean_travel_times)), str(np.mean(self.episode_mean_time_losses))])

        # Sums/totals
        string_array_to_log.append(
            ["Sums", str(np.sum(self.episode_mean_travel_times)), str(np.sum(self.episode_mean_time_losses))])

        logger.logtxt(string_array_to_log, "travel_time_and_time_loss")


