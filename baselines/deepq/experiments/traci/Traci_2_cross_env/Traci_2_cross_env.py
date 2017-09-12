import logging
import os
import sys
from collections import deque

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from threading import Thread
import copy

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

logger = logging.getLogger(__name__)


class UniqueCounter:
    def __init__(self):
        self.ids = {}

    def add_many(self, list_of_ids):
        for i in list_of_ids:
            self.ids[i] = True

    def remove_many(self, list_of_ids):
        for i in list_of_ids:
            if i in self.ids:
                del self.ids[i]

    def get_count(self):
        return len(self.ids)


class Traci_2_cross_env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50,
        "traci": True
    }

    def generate_routefile(self):
        """ Generates an XML file that gets loaded by the simulator which spawns all the cars.
            Cars MUST HAVE UNIQUE ID
        """

        N = 300  # number of time steps
        # demand per second from different directions
        p_w_e = 1/10
        p_e_w = 1/10
        p_n_s_a = 1/10
        p_s_n_a = 1/10
        p_n_s_b = 1/10
        p_s_n_b = 1/10
        with open("scenarios/2_cross/cross.rou.xml", "w") as routes:
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
        self.route_file_generated = True

    def __traci_start__(self):
        traci.start([self.sumo_binary, "-c", "scenarios/2_cross/cross.sumocfg", "--tripinfo-output", "tripinfo.xml", "--start",
                     "--quit-on-end"])

    def __init__(self):
        self.shouldRender=False
        self.num_actions=9
        self.num_state_scalars=10
        self.num_history_states=4
        self.min_state_scalar_value = 0
        self.max_state_scalar_value=20

        self.restart()

    def restart(self):
        self.generate_routefile()
        if self.shouldRender:
            self.sumo_binary = checkBinary('sumo-gui')
        else:
            self.sumo_binary = checkBinary('sumo')
        Thread(target=self.__traci_start__())

        self.max_cars_in_queue = 20
        self.route_file_generated = False
        self.num_inductors = 4
        self.vehicle_ids = []

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(self.min_state_scalar_value, self.max_state_scalar_value,shape=(self.num_state_scalars*self.num_history_states))
        # self.reward_range = (-4*max_cars_in_queue, 4*max_cars_in_queue)

        self.unique_counters = [UniqueCounter() for _ in range(self.num_inductors*2)]

        self.state = deque([], maxlen=self.num_history_states)
        for i in range(self.num_history_states):
            self.state.append(np.zeros(self.num_state_scalars))
        self._seed()
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def discrete_to_multidiscrete(self, action, num_actions):
        mod_rest = action % num_actions
        div_floor = action // num_actions
        return [mod_rest, div_floor]

    def set_light_phase(self,light_id,action,cur_phase):
        # Run action

        if action == 0:
            if cur_phase == 2:
                traci.trafficlights.setPhase(light_id, 3)
            elif cur_phase == 0:
                traci.trafficlights.setPhase(light_id, 0)
        elif action == 1:
            if cur_phase == 0:
                traci.trafficlights.setPhase(light_id, 1)
            elif cur_phase == 2:
                traci.trafficlights.setPhase(light_id, 2)
        else:
            pass  # do nothing

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # convert action into two actions
        action = self.discrete_to_multidiscrete(action=action, num_actions=3)
        phase_a = traci.trafficlights.getPhase("a")
        phase_b = traci.trafficlights.getPhase("b")
        self.set_light_phase("a",action[0],phase_a)
        self.set_light_phase("b",action[1],phase_b)

        # Run simulation step
        traci.simulationStep()

        # Build state
        cur_state=np.zeros(self.num_state_scalars)
        for i in range(self.num_inductors):
            input_id = "i" + str(i)
            output_id = "o" + str(i)

            cars_on_in_inductor = traci.inductionloop.getLastStepVehicleIDs(input_id+"_a")
            cars_on_out_inductor = traci.inductionloop.getLastStepVehicleIDs(output_id+"_a")

            self.unique_counters[i].add_many(cars_on_in_inductor)
            self.unique_counters[i].remove_many(cars_on_out_inductor)

            cars_on_in_inductor = traci.inductionloop.getLastStepVehicleIDs(input_id + "_b")
            cars_on_out_inductor = traci.inductionloop.getLastStepVehicleIDs(output_id + "_b")

            self.unique_counters[i+self.num_inductors].add_many(cars_on_in_inductor)
            self.unique_counters[i+self.num_inductors].remove_many(cars_on_out_inductor)


            num_cars_a = self.unique_counters[i].get_count()
            num_cars_b=self.unique_counters[i+self.num_inductors].get_count()
            # Add one if car on inductor
            if traci.inductionloop.getLastStepOccupancy(output_id + "_a") !=-1:
                num_cars_a+=1
            if traci.inductionloop.getLastStepOccupancy(output_id + "_b") != -1:
                num_cars_b += 1

            cur_state[i] = min(num_cars_a, self.max_cars_in_queue)
            cur_state[i+self.num_inductors] = min(num_cars_b, self.max_cars_in_queue)

        cur_state[8] = phase_a
        cur_state[9] = phase_b


        self.state.append(cur_state)

        # Build reward
        reward = self.reward_total_waiting_vehicles()

        # See if done
        done = traci.simulation.getMinExpectedNumber() < 1

        return np.hstack(self.state), reward, done, {}

    def reward_leaving_cars(self):
        s = 0
        for i in range(4):
            leaving_id = "l" + str(i)
            s += traci.inductionloop.getLastStepVehicleNumber(leaving_id)
        return s

    def reward_emission(self):
        self.vehicle_ids = traci.vehicle.getIDList()
        emissions = []
        for veh_id in self.vehicle_ids:
            emissions.append(traci.vehicle.getCO2Emission(veh_id))
        return -np.mean(emissions)

    def reward_total_waiting_vehicles(self):
        self.vehicle_ids = traci.vehicle.getIDList()
        total_wait_time = 0.0
        for veh_id in self.vehicle_ids:
            if traci.vehicle.getSpeed(veh_id) < 1:
                total_wait_time += 1.0
        return -total_wait_time

    def reward_total_in_queue(self):
        return -sum(self.state)

    def reward_squared_wait_sum(self):
        self.vehicle_ids = traci.vehicle.getIDList()
        wait_sum = 0
        for veh_id in self.vehicle_ids:
            wait_sum += traci.vehicle.getWaitingTime(veh_id)
        return -np.mean(np.square(wait_sum))

    def _reset(self):
        # Check if actually done, might be initial reset call
        if traci.simulation.getMinExpectedNumber() < 1:
            traci.close(wait=False)
            self.restart()
        return np.hstack(self.state)

    def _render(self, mode='human', close=False):
        self.shouldRender = True
        print("Render not implemented. Set sumo_binary = checkBinary('sumo-gui')")
