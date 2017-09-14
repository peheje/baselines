import sys
import os
import gym

import traci
import numpy as np

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


class BaseTraciEnv(gym.Env):
    def _reset(self):
        raise NotImplementedError()
        pass

    def _step(self, action):
        raise NotImplementedError()
        pass

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
        return -np.mean(emissions)

    def reward_total_in_queue(self, state):
        raise NotImplementedError("state contains traffic lights, please implement this function")
        return -sum(state)

    @staticmethod
    def reward_squared_wait_sum():
        vehs = traci.vehicle.getIDList()
        wait_sum = 0
        for veh_id in vehs:
            wait_sum += traci.vehicle.getWaitingTime(veh_id)
        return -np.mean(np.square(wait_sum))

    @staticmethod
    def discrete_to_multidiscrete(action, num_actions):
        mod_rest = action % num_actions
        div_floor = action // num_actions
        return [mod_rest, div_floor]

    @staticmethod
    def set_light_phase(light_id, action, cur_phase):
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
