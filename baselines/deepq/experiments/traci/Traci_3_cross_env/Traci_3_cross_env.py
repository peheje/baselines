import logging
import os
import sys
from collections import deque
from threading import Thread
import tempfile
import gym
import numpy as np
from BaseTraciEnv import BaseTraciEnv
from utilities.UniqueCounter import UniqueCounter
from gym import spaces
from gym.utils import seeding

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
        tos = ["Ae", "Be", "Ce", "De", "Ee", "Ge", "He", "Ie", "Je"]
        paths = []

        for f in froms:
            for t in tos:
                if not t.startswith(f[0]):
                    paths.append((f, t))

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as routes:
            self.route_file_name = routes.name
            print("<routes>", file=routes)
            print(
                '<vType id="carType" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>',
                file=routes)
            for i, p in enumerate(paths):
                print('<route id="route{}" edges="{} {}" />'.format(i, p[0], p[1]), file=routes)

            vehid = 0
            for i in range(self.num_car_chances):
                print('<vehicle id="{}" type="carType" route="route{}" depart="{}"/>'.format(vehid, random_int(0, len(paths)), i), file=routes)
                vehid += 1

            print("</routes>", file=routes)

        print("have created spawn_cars at", self.route_file_name)

        #traci.route.add("trip", ["Bs", "hr_south_out", "gneE3", "mc_north_in"])
        #for i in range(100):
        #    traci.vehicle.add("car" + str(i), "trip", typeID="reroutingType")

    def __traci_start__(self):
        traci.start(
            [self.sumo_binary, "-c", "scenarios/3_cross/cross.sumocfg", "--tripinfo-output", self.tripinfo_file_name,
             "--start",
             "--quit-on-end"])

    def __init__(self):
        # Start by calling parent init
        BaseTraciEnv.__init__(self)
        self.route_file_name = None
        self.should_render = False
        self.num_actions = 2 ** 4
        self.num_state_scalars = 4 * 4 + 4
        self.num_history_states = 4
        self.max_cars_in_queue = 20
        self.min_state_scalar_value = 0
        self.max_state_scalar_value = 20
        self.sumo_binary = None
        self.e3ids = None
        self.state = []
        self.unique_counters = []

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

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(self.min_state_scalar_value, self.max_state_scalar_value,
                                            shape=(self.num_state_scalars * self.num_history_states))
        # self.reward_range = (-4*max_cars_in_queue, 4*max_cars_in_queue)

        self.state = deque([], maxlen=self.num_history_states)
        for i in range(self.num_history_states):
            self.state.append(np.zeros(self.num_state_scalars))
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # convert action into two actions
        action = self.discrete_to_multidiscrete_4cross(action)
        # self.set_light_phase("a", action[0], phase_a)
        # self.set_light_phase("b", action[1], phase_b)

        # Run simulation step
        traci.simulationStep()

        # Build state
        # self.state = deque([], maxlen=self.num_history_states)
        if self.e3ids is None:
            self.e3ids = traci.multientryexit.getIDList()
        cur_state = []
        for id in self.e3ids:
            cur_state.append(len(traci.multientryexit.getLastStepVehicleIDs(id)))
        cur_state = cur_state + self.get_traffic_states()
        self.state.append(cur_state)

        print("STATE", self.state)

        # Build reward
        reward = self.reward_func()

        # See if done
        done = traci.simulation.getMinExpectedNumber() < 1

        self.log_end_step(reward)

        return np.hstack(self.state), reward, done, {}

    def get_traffic_states(self):
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
        return np.hstack(self.state)

    def _render(self, mode='human', close=False):
        self.should_render = True
        self.restart()
        print("Render not implemented! Set sumo_binary = checkBinary('sumo-gui')")