import logging
import os
import sys
from collections import deque, OrderedDict
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
            for iter, f in enumerate(froms):
                # If spawning from big road
                if any(bigroad in f for bigroad in big_roads):
                    spawn_prob = self.car_props[0]
                else:
                    spawn_prob = self.car_props[1]

                print('<flow id="{}" from="{}" begin="0" end="{}" probability="{}"/>'.format(iter, f,
                                                                                             self.num_car_chances,
                                                                                             spawn_prob), file=flows)
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
                                         " --accept-all-destinations", shell=True)

        print(status)

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
             "--time-to-teleport", "-1",
             "--route-files", self.route_file_name])

    def __init__(self):
        # Start by calling parent init
        BaseTraciEnv.__init__(self)
        self.flow_file_name = None
        self.should_render = False
        self.num_actions = 2 ** 4
        self.num_history_state_scalars = None  # This gets calculated later
        self.num_nonhistory_state_scalars=None
        self.min_state_scalar_value = 0
        self.max_state_scalar_value = 1000
        self.sumo_binary = None
        self.e3ids = None
        self.state = []
        self.unique_counters = []
        self.route_file_name = None

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
        self.num_nonhistory_state_scalars=self.calculate_num_nonhistory_state_scalars()
        self.total_num_state_scalars=(self.num_history_state_scalars * self.num_history_states) + self.num_nonhistory_state_scalars
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(self.min_state_scalar_value, self.max_state_scalar_value,
                                            shape=(self.total_num_state_scalars))

        self.state = deque([], maxlen=self.num_history_states)
        for i in range(self.num_history_states):
            self.state.append(np.zeros(self.num_history_state_scalars))
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # convert action into many actions
        action = self.discrete_to_multidiscrete_4cross(action)
        for i, tlsid in enumerate(traci.trafficlights.getIDList()):
            phase = traci.trafficlights.getPhase(tlsid)
            self.set_light_phase_4_cross(tlsid, action[i], phase)

        # Run simulation step
        traci.simulationStep()

        # Build state
        total_state = self.get_state_multientryexit()

        # Build reward
        reward = self.reward_func()

        # See if done
        done = traci.simulation.getMinExpectedNumber() < 1

        self.log_end_step(reward)

        return total_state, reward, done, {}

    @staticmethod
    def get_traffic_states():
        ids = []
        for tid in traci.trafficlights.getIDList():
            ids.append(traci.trafficlights.getPhase(tid))
        return ids

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
