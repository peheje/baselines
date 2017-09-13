import logging
import os
import sys
from threading import Thread

import gym
import numpy as np
from experiments.traci.utilities.UniqueCounter import UniqueCounter
from gym import spaces
from gym.utils import seeding

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


class Traci_1_cross_env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50,
        "traci": True
    }

    def generate_routefile(self):
        """ Generates an XML file that gets loaded by the simulator which spawns all the cars.
            Cars MUST HAVE UNIQUE ID
        """

        N = 1000  # number of time steps
        # demand per second from different directions
        p_w_e = 1 / 10
        p_e_w = 1 / 10
        p_n_s = 1 / 10
        p_s_n = 1 / 10
        with open("scenarios/1_cross/cross.rou.xml", "w") as routes:
            print("""<routes>
            <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
            <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
            <route id="right" edges="51o 1i 2o 52i" />
            <route id="left" edges="52o 2i 1o 51i" />
            <route id="up" edges="53o 3i 4o 54i" />
            <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
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
                if np.random.uniform() < p_n_s:
                    print(
                        '    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (vehNr, i),
                        file=routes)
                    vehNr += 1
                if np.random.uniform() < p_s_n:
                    print(
                        '    <vehicle id="up_%i" type="typeNS" route="up" depart="%i" color="1,0,0"/>' % (vehNr, i),
                        file=routes)
                    vehNr += 1
            print("</routes>", file=routes)
        self.route_file_generated = True

    def __traci_start__(self):
        traci.start(
            [self.sumo_binary, "-c", "scenarios/1_cross/cross.sumocfg", "--tripinfo-output", "tripinfo.xml", "--start",
             "--quit-on-end"])

    def __init__(self):
        self.shouldRender = True
        self.restart()

    def restart(self):
        self.generate_routefile()
        if self.shouldRender:
            self.sumo_binary = checkBinary('sumo-gui')
        else:
            self.sumo_binary = checkBinary('sumo')
        Thread(target=self.__traci_start__())

        self.max_cars_in_queue = 20
        self.traffic_phases = 4
        high = np.array([self.max_cars_in_queue, self.max_cars_in_queue,
                         self.max_cars_in_queue, self.max_cars_in_queue, self.traffic_phases])
        low = np.array([0, 0, 0, 0, 0])

        self.route_file_generated = False
        self.num_inductors = 4
        self.vehicle_ids = []

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high)
        # self.reward_range = (-4*max_cars_in_queue, 4*max_cars_in_queue)

        self.unique_counters = [
            UniqueCounter(),
            UniqueCounter(),
            UniqueCounter(),
            UniqueCounter()
        ]

        self.state = [0, 0, 0, 0, 0]
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Run action
        phase = traci.trafficlights.getPhase("0")
        if action == 0:
            if phase == 2:
                traci.trafficlights.setPhase("0", 3)
            elif phase == 0:
                traci.trafficlights.setPhase("0", 0)
        elif action == 1:
            if phase == 0:
                traci.trafficlights.setPhase("0", 1)
            elif phase == 2:
                traci.trafficlights.setPhase("0", 2)
        else:
            pass  # do nothing

        # Run simulation step
        traci.simulationStep()

        # Build state
        for i in range(self.num_inductors):
            input_id = "i" + str(i)
            output_id = "o" + str(i)

            cars_on_in_inductor = traci.inductionloop.getLastStepVehicleIDs(input_id)
            cars_on_out_inductor = traci.inductionloop.getLastStepVehicleIDs(output_id)

            self.unique_counters[i].add_many(cars_on_in_inductor)
            self.unique_counters[i].remove_many(cars_on_out_inductor)

        for i in range(self.num_inductors):
            self.state[i] = min(self.unique_counters[i].get_count(), self.max_cars_in_queue)
        self.state[4] = phase

        # Build reward
        reward = self.reward_total_waiting_vehicles()

        # See if done
        done = traci.simulation.getMinExpectedNumber() < 1

        return np.array(self.state), reward, done, {}

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
        return np.array([0, 0, 0, 0, 0])

    def _render(self, mode='human', close=False):
        self.shouldRender = True
        print("Render not implemented. Set sumo_binary = checkBinary('sumo-gui')")
