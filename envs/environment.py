import math

import numpy as np
from utils.setting_setup import *
import scipy

from envs.env_utils import *
from envs.env_agent_utils import *


class base_env(rsma_utils, env_agent_utils):
    def __init__(self, args):
        # Network setting


        # Power setting


        # Base station initialization
        self.BS_x = 0
        self.BS_y = 0
        self.BS_R_Range = 1
        self.BS_R_min = 0.1

        """ ========================================= """
        """ ===== Function-based Initialization ===== """
        """ ========================================= """
        self.BS_location = np.expand_dims(self._location_BS_Generator(), axis=0)
        self.U_location = self._location_CU_Generator()
        self.User_trajectory = self._trajectory_U_Generator()
        self.distance_CU_BS = self._distance_Calculated(self.U_location, self.BS_location)

        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)
        self.commonDataRate = self._calculateDataRate(self.ChannelGain)
        self.T = 0  # initialize rewards)

        """ ============================ """
        """     Environment Settings     """
        """ ============================ """
        self.rewardMatrix = np.array([])
        self.observation_space = self._wrapState().squeeze()
        self.action_space = self._wrapAction()

    def step(self, action, step):
        """     Environment change      """
        self.User_trajectory = np.expand_dims(self._trajectory_U_Generator(), axis=0)
        self.U_location = self.User_trajectory + self.U_location

        """     Re-calculate channel gain     """
        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)
        state_next = self._wrapState()
        """     Reward      """
        reward = None

        """     info|done      """
        if step == self.max_step:
            done = True
        else:
            done = False

        info = None
        return state_next, reward, done, info

    def reset(self):
        # Base station initialization
        self.BS_location = np.expand_dims(self._location_BS_Generator(), axis=0)

        """     Use initialization      """
        # self.U_location = np.expand_dims(self._location_CU_Generator(), axis=0)
        # self.User_trajectory = np.expand_dims(self._trajectory_U_Generator(), axis=0)
        self.U_location = self._location_CU_Generator()
        self.User_trajectory = self._trajectory_U_Generator()
        # Distance calculation
        self.distance_CU_BS = self._distance_Calculated(self.BS_location, self.U_location)

        # re-calculate channel gain
        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)

        # Generate next state [set of ChannelGain]
        state_next = self._wrapState()
        return state_next

    def close(self):
        pass


if __name__ == '__main__':
    args = get_arguments()
    env = base_env(args)
