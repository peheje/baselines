import json
import logging
import os
import sys
from collections import deque, OrderedDict
from pathlib import Path
from shutil import copyfile
from threading import Thread
import tempfile
import gym
import numpy as np
from BaseTraciEnv import BaseTraciEnv
from utilities.UniqueCounter import UniqueCounter
from gym import spaces
from gym.utils import seeding
import subprocess

import traci
from sumolib import checkBinary

logger = logging.getLogger(__name__)


def random():
    return np.random.uniform()


def random_int(low, high):
    return np.random.randint(low, high)


class Traci_3_cross_env(BaseTraciEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50,
        "traci": True
    }

    def spawn_cars(self):
        froms = ["As", "Bs", "Cs", "Ds", "Es", "Fs", "Gs", "Hs", "Is", "Js"]
        big_roads = ["A", "B", "C", "H", "I", "J"]

        # Flow files says for all incoming lanes the probability of spawning a car each timestep
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as flows:
            self.flow_file_name = flows.name
            print("<routes>", file=flows)

            if self.enjoy_car_probs:
                hard_coded_bigroad_probs = [0.25,
                                            0.50,
                                            1.0,
                                            0.75,
                                            0.25,
                                            0.50,
                                            1.0,
                                            0.50,
                                            0.25,
                                            0.25]
                increment_every_interval = self.num_car_chances // len(hard_coded_bigroad_probs)

                flow_id = 0
                cur_interval_start = 0
                for interval in range(len(hard_coded_bigroad_probs)):
                    for iter, f in enumerate(froms):
                        # If spawning from big road
                        if any(bigroad in f for bigroad in big_roads):
                            spawn_prob = hard_coded_bigroad_probs[interval]
                        else:
                            spawn_prob = hard_coded_bigroad_probs[interval] / 10

                        print('<flow id="{}" from="{}" begin="{}" end="{}" probability="{}"/>'.format(flow_id, f,
                                                                                                      cur_interval_start,
                                                                                                      cur_interval_start + increment_every_interval,
                                                                                                      spawn_prob),
                              file=flows)
                        flow_id += 1
                    cur_interval_start += increment_every_interval

            else:
                for iter, f in enumerate(froms):
                    # If spawning from big road
                    if any(bigroad in f for bigroad in big_roads):
                        spawn_prob = self.car_probabilities[0]
                    else:
                        spawn_prob = self.car_probabilities[1]

                    print('<flow id="{}" from="{}" begin="0" end="{}" probability="{}"/>'.format(iter, f,
                                                                                                 self.num_car_chances,
                                                                                                 spawn_prob),
                          file=flows)
            print("</routes>", file=flows)

        # Make temp file for routes
        temp_route_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        self.route_file_name = temp_route_file.name
        print("TMP ROUTE FILE PATH", self.route_file_name)

        # Creates routes from incoming cars from turn probabilities on the intersections
        status = subprocess.check_output("jtrrouter" +
                                         " -n scenarios/3_cross/randersvej.net.xml" +
                                         " -f {}".format(self.flow_file_name) +
                                         " -o {}".format(self.route_file_name) +
                                         " --turn-ratio-files scenarios/3_cross/turn_probs" +
                                         " --turn-defaults 20,70,10" +
                                         " --seed " + str(self.jtrroute_seed) +
                                         " --accept-all-destinations", shell=True)

        print(status)
        self.jtrroute_seed += 1

        # Run webster on route file (tls only used if not controlled)
        status = subprocess.check_output("python3 utilities/tlsCycleAdaptation_timestep_fix.py" +
                                         " -n scenarios/3_cross/randersvej.net.xml" +
                                         " -r {}".format(self.route_file_name) +
                                         " -o scenarios/3_cross/webster.tls.xml" +
                                         " --program c" +
                                         " --timestep {}".format(self.num_car_chances),
                                         shell=True)
        print(status)

    def __traci_start__(self):
        traci.start(
            [self.sumo_binary,
             "-c", "scenarios/3_cross/cross.sumocfg",
             "--tripinfo-output", self.tripinfo_file_name,
             "--start",
             "--quit-on-end",
             "--time-to-teleport", "300",
             "--route-files", self.route_file_name])

    def __init__(self):
        # Start by calling parent init
        BaseTraciEnv.__init__(self)
        self.flow_file_name = None
        self.should_render = False
        self.num_actions = None
        self.num_history_state_scalars = None  # This gets calculated later
        self.num_nonhistory_state_scalars = None
        self.num_trafficlights = 4
        self.num_history_states = None
        self.min_state_scalar_value = 0
        self.max_state_scalar_value = 1000
        self.sumo_binary = None
        self.e3ids = None
        self.state = []
        self.unique_counters = []
        self.route_file_name = None
        self.jtrroute_seed = 0

    def get_state_multientryexit(self):

        if not self.subscribed_to_multientryexit and len(self.multientryexit_subscriptions_ids) > 0:
            for mee_id in traci.multientryexit.getIDList():
                traci.multientryexit.subscribe(mee_id, self.multientryexit_subscriptions_ids)
            self.subscribed_to_multientryexit = True

        raw_mee_state = traci.multientryexit.getSubscriptionResults()

        subscription_state = []
        if self.state_use_num_cars_in_queue_history:
            subscription_state += self.extract_list(raw_mee_state, traci.constants.LAST_STEP_VEHICLE_NUMBER)
        if self.state_use_avg_speed_between_detectors_history:
            subscription_state += self.extract_list(raw_mee_state, traci.constants.LAST_STEP_MEAN_SPEED)
        if self.state_use_tl_state_history:
            subscription_state += self.get_traffic_states()

        self.state.append(np.array(subscription_state, dtype=float))

        state_to_return = np.hstack(self.state)
        if self.state_use_time_since_tl_change:
            state_to_return = np.concatenate([state_to_return, list(self.time_since_tl_change.values())])

        # old = self.get_state_multientryexit_old()
        # old_json = json.dumps(old.tolist())
        # new_json = json.dumps(state_to_return.tolist())
        # print("old", old_json)
        # print("new", new_json)
        # assert old_json == new_json

        return state_to_return

    def get_traffic_states(self):
        if not self.subscribed_to_trafficlights and len(self.trafficlights_subscriptions_ids) > 0:
            for tid in traci.trafficlights.getIDList():
                traci.trafficlights.subscribe(tid, self.trafficlights_subscriptions_ids)
            self.subscribed_to_trafficlights = True

        raw_til_state = traci.trafficlights.getSubscriptionResults()
        phases = self.extract_list(raw_til_state, traci.constants.TL_CURRENT_PHASE)
        return phases

    def restart(self):
        self.spawn_cars()
        if self.should_render:
            self.sumo_binary = checkBinary('sumo-gui')
        else:
            self.sumo_binary = checkBinary('sumo')
        try:
            traci.close(wait=True)
        except:
            pass
        Thread(target=self.__traci_start__())

        self.num_history_state_scalars = self.calculate_num_history_state_scalars()
        self.num_nonhistory_state_scalars = self.calculate_num_nonhistory_state_scalars()
        self.total_num_state_scalars = (
                                       self.num_history_state_scalars * self.num_history_states) + self.num_nonhistory_state_scalars
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(self.min_state_scalar_value, self.max_state_scalar_value,
                                            shape=(self.total_num_state_scalars))

        self.state = deque([], maxlen=self.num_history_states)
        self.old_state = deque([], maxlen=self.num_history_states)
        for i in range(self.num_history_states):
            self.state.append(np.zeros(self.num_history_state_scalars))
            self.old_state.append(np.zeros(self.num_history_state_scalars))
        self._seed()

        # Getting IDS for static content
        self.trafficlights_ids = traci.trafficlights.getIDList()

        # Subscriptions
        self.subscribed_to_multientryexit = False
        self.subscribed_to_trafficlights = False

        self.multientryexit_subscriptions_ids = []
        if self.state_use_num_cars_in_queue_history:
            self.multientryexit_subscriptions_ids.append(traci.constants.LAST_STEP_VEHICLE_NUMBER)
        if self.state_use_avg_speed_between_detectors_history:
            self.multientryexit_subscriptions_ids.append(traci.constants.LAST_STEP_MEAN_SPEED)

        self.trafficlights_subscriptions_ids = []
        if self.state_use_tl_state_history:
            self.trafficlights_subscriptions_ids.append(traci.constants.TL_CURRENT_PHASE)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if self.perform_actions:
            # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
            # convert action into many actions
            action = self.action_converter(action)

            phases = self.get_traffic_states()
            for i, tlsid in enumerate(self.trafficlights_ids):
                # phase = traci.trafficlights.getPhase(tlsid)
                # assert phases[i] == phase
                self.set_light_phase_4_cross(tlsid, action[i], phases[i])

        # Run simulation step
        traci.simulationStep()

        # Build state
        total_state = 0
        if self.perform_actions:
            total_state = self.get_state_multientryexit()

        # Build reward
        reward = self.reward_func()

        # See if done
        done = traci.simulation.getMinExpectedNumber() < 1

        self.log_end_step(reward)

        return total_state, reward, done, {}

    def _reset(self):
        # Check if actually done, might be initial reset call
        if traci.simulation.getMinExpectedNumber() < 1:
            traci.close(wait=True)  # Wait for tripinfo to be written
            self.log_end_episode(0)
            BaseTraciEnv._reset(self)
            self.restart()
        return np.zeros(self.total_num_state_scalars)

    def _render(self, mode='human', close=False):
        if not close:
            self.should_render = True
            self.restart()
