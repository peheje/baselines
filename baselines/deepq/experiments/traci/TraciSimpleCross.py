import logging
import os
import sys

import gym
import numpy as np
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


class TraciSimpleCross(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def generate_routefile(self):
        N = 3600  # number of time steps
        # demand per second from different directions
        p_w_e = 1.0 / 10
        p_e_w = 1.0 / 11
        p_n_s = 1.0 / 30
        with open("data/cross.rou.xml", "w") as routes:
            print("""<routes>
            <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
            <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>
            <route id="right" edges="51o 1i 2o 52i" />
            <route id="left" edges="52o 2i 1o 51i" />
            <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
            vehNr = 0
            for i in range(N):
                if np.random.uniform() < p_w_e:
                    print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (vehNr, i), file=routes)
                    vehNr += 1
                if np.random.uniform() < p_e_w:
                    print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (vehNr, i), file=routes)
                    vehNr += 1
                if np.random.uniform() < p_n_s:
                    print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (vehNr, i), file=routes)
                    vehNr += 1
            print("</routes>", file=routes)
        self.route_file_generated = True

    def __init__(self):
        self.max_cars_in_queue = 1000
        high = np.array([self.max_cars_in_queue, self.max_cars_in_queue,
                         self.max_cars_in_queue, self.max_cars_in_queue])

        self.route_file_generated = False
        self.num_inductors = 4
        self.vehicle_ids = traci.getIDList()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)
        # self.reward_range = (-4*max_cars_in_queue, 4*max_cars_in_queue)

        self.state = [0, 0, 0, 0]
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Run simulation step
        traci.simulationStep()

        # Build state
        for inductor_id in range(self.num_inductors):
            self.state[inductor_id] += traci.getLastStepVehicleNumber("i"+str(inductor_id))
        for inductor_id in range(self.num_inductors):
            self.state[inductor_id] -= traci.getLastStepVehicleNumber("o"+str(inductor_id))

        # Build reward
        wait_sum = 0
        for veh_id in self.vehicle_ids:
            wait_sum += traci.getWaitingTime(veh_id)

        # See if done
        done = traci.simulation.getMinExpectedNumber() < 1

        return np.array(self.state), np.mean(wait_sum), done, {}

    def _reset(self):

        traci.close()

        # this script has been called from the command line. It will start sumo as a
        # server, then connect and run
        # sumo_binary = checkBinary('sumo')
        sumo_binary = checkBinary('sumo-gui')

        # first, generate the route file for this simulation
        if not self.route_file_generated:
            self.generate_routefile()

        self.vehicle_ids = traci.getIDList()

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([sumo_binary, "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])

        return np.array([0, 0, 0, 0])

    def _render(self, mode='human', close=False):
        print("Render not implemented. Set sumo_binary = checkBinary('sumo-gui')")
