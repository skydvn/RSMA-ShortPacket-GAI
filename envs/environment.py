import math

import numpy as np
from utils.setting_setup import *
import scipy

from envs.env_utils import *
from envs.env_agent_utils import *


class base_env(rsma_utils, env_agent_utils):
    def __init__(self, args):
        # Base station initialization
        self.BS_x = 0
        self.BS_y = 0
        self.BS_R_Range = 1
        self.BS_R_min = 0.1

        """ ========================================= """
        """ =============     Network    ============ """
        """ ========================================= """
        self.antenna_num = args.antenna_num # Number of antenna setting
        self.user_num = args.user_num       # Number of users
        self.bandwidth = self.bandwidth     # Bandwidth in (MHz)
        self.NFdB = args.NFdB               # From 280 GHz ~ 330 GHz

        self.PTdB = np.arange(0,20,2.5)
        self.PTdB1 = np.arange(min(self.PTdB),max(self.PTdB),1)

        self.sig2dBm = -174 + 10*np.log10(self.bandwidth) + self.NFdB   # Noise power (dBm)
        # Carrier Frequency (GHz) - 300 GHz
        self.carrier_freq = args.carrier_freq
        # Absorption Loss Coefficient Measured at 300 Ghz
        self.kabs = args.kabs
        # Antenna Gain of Tx
        # Antenna Gain of Tx

        # Radius of cell-in
        # Radius of cell-out

        # Power allocation
        # Power allocation

        # Rate common
        # Rate private

        # Threshold common
        # Threshold private

        # Imperfect SIC coefficient


        """ ========================================= """
        """ ========== Channel Realization ========== """
        """ ========================================= """
        # Channel Realization

        # Random user-k location

        # LSF * MAP part of channel hk

        # Power setting

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
