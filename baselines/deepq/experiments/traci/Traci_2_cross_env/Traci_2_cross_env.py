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


class Traci_2_cross_env(BaseTraciEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50,
        "traci": True
    }

    def generate_routefile(self):
        """ Generates an XML file that gets loaded by the simulator which spawns all the cars.
            Cars MUST HAVE UNIQUE ID
        """

        N = self.num_car_chances  # number of time steps
        # demand per second from different directions
        p_w_e = self.car_props[0]
        p_e_w = self.car_props[1]
        p_n_s_a = self.car_props[2]
        p_s_n_a = self.car_props[3]
        p_n_s_b = self.car_props[4]
        p_s_n_b = self.car_props[5]
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as routes:
            self.route_file_name=routes.name
            print("""<routes>
            <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
            <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
            <route id="right" edges="51o 1i_a 2o 1o_b 52o" />
            <route id="left" edges="52i 1i_b 2i 1o_a 51i" />
            <route id="up_a" edges="53o 3i_a 4o_a 54i" />
            <route id="down_a" edges="54o 4i_a 3o_a 53i" />
            <route id="up_b" edges="3i_b 4o_b" />
            <route id="down_b" edges="4i_b 3o_b" />""", file=routes)
            vehNr = 0
            for i in range(N):
                if np.random.uniform() < p_w_e:
                    print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (vehNr, i),
                          file=routes)
                    vehNr += 1
                if np.random.uniform() < p_e_w:
                    print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (vehNr, i),
                          file=routes)
                    vehNr += 1
                if np.random.uniform() < p_n_s_a:
                    print(
                        '    <vehicle id="down_%i" type="typeNS" route="down_a" depart="%i"/>' % (vehNr, i),
                        file=routes)
                    vehNr += 1
                if np.random.uniform() < p_s_n_a:
                    print(
                        '    <vehicle id="up_%i" type="typeNS" route="up_a" depart="%i"/>' % (vehNr, i),
                        file=routes)
                    vehNr += 1
                if np.random.uniform() < p_n_s_b:
                    print(
                        '    <vehicle id="up_%i" type="typeNS" route="up_b" depart="%i"/>' % (vehNr, i),
                        file=routes)
                    vehNr += 1
                if np.random.uniform() < p_s_n_b:
                    print(
                        '    <vehicle id="up_%i" type="typeNS" route="down_b" depart="%i"/>' % (vehNr, i),
                        file=routes)
                    vehNr += 1
            print("</routes>", file=routes)

    def __traci_start__(self):
        traci.start(
            [self.sumo_binary, "-c", "scenarios/2_cross/cross.sumocfg", "--tripinfo-output", self.tripinfo_file_name, "--start",
             "--quit-on-end","--route-files",self.route_file_name])

    def __init__(self):
        #Start by calling parent init
        BaseTraciEnv.__init__(self)
        self.num_queues_pr_traffic = 4
        self.shouldRender = False

        self.num_actions_pr_trafficlight = 3
        self.num_trafficlights = 2

        self.num_actions = self.num_actions_pr_trafficlight ** self.num_trafficlights

        self.num_state_scalars = 10
        self.num_history_states = 4
        self.max_cars_in_queue = 20
        self.min_state_scalar_value = 0
        self.max_state_scalar_value = 20
        self.sumo_binary = None
        self.state = []
        self.unique_counters = []



    def restart(self):
        self.generate_routefile()
        if self.shouldRender:
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

        self.unique_counters = [UniqueCounter() for _ in range(self.num_queues_pr_traffic * 2)]

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
        action = self.discrete_to_multidiscrete_2cross(action=action, num_actions=self.num_actions_pr_trafficlight)
        phase_a = traci.trafficlights.getPhase("a")
        phase_b = traci.trafficlights.getPhase("b")
        self.set_light_phase("a", action[0], phase_a)
        self.set_light_phase("b", action[1], phase_b)

        # Run simulation step
        traci.simulationStep()

        # Build state
        cur_state = np.zeros(self.num_state_scalars)
        for i in range(self.num_queues_pr_traffic):
            input_id = "i" + str(i)
            output_id = "o" + str(i)

            cars_on_in_inductor = traci.inductionloop.getLastStepVehicleIDs(input_id + "_a")
            cars_on_out_inductor = traci.inductionloop.getLastStepVehicleIDs(output_id + "_a")

            self.unique_counters[i].add_many(cars_on_in_inductor)
            self.unique_counters[i].remove_many(cars_on_out_inductor)

            cars_on_in_inductor = traci.inductionloop.getLastStepVehicleIDs(input_id + "_b")
            cars_on_out_inductor = traci.inductionloop.getLastStepVehicleIDs(output_id + "_b")

            self.unique_counters[i + self.num_queues_pr_traffic].add_many(cars_on_in_inductor)
            self.unique_counters[i + self.num_queues_pr_traffic].remove_many(cars_on_out_inductor)

            num_cars_a = self.unique_counters[i].get_count()
            num_cars_b = self.unique_counters[i + self.num_queues_pr_traffic].get_count()
            # Add one if car on inductor
            if traci.inductionloop.getLastStepOccupancy(output_id + "_a") != -1:
                num_cars_a += 1
            if traci.inductionloop.getLastStepOccupancy(output_id + "_b") != -1:
                num_cars_b += 1

            cur_state[i] = min(num_cars_a, self.max_cars_in_queue)
            cur_state[i + self.num_queues_pr_traffic] = min(num_cars_b, self.max_cars_in_queue)

        cur_state[8] = phase_a
        cur_state[9] = phase_b

        self.state.append(cur_state)

        # Build reward
        reward = self.reward_func()

        # See if done
        done = traci.simulation.getMinExpectedNumber() < 1

        self.log_end_step(reward)

        return np.hstack(self.state), reward, done, {}

    def _reset(self):
        # Check if actually done, might be initial reset call
        if traci.simulation.getMinExpectedNumber() < 1:
            traci.close(wait=True) # Wait for tripinfo to be written
            self.log_end_episode(0)
            BaseTraciEnv._reset(self)
            self.restart()
        return np.hstack(self.state)

    def _render(self, mode='human', close=False):
        if not close:
            self.shouldRender = True
            self.restart()
