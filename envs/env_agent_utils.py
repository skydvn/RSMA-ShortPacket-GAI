import numpy as np
from utils.setting_setup import *
import scipy


class env_agent_utils:
    def __init__(self):
        pass

    def _wrapState(self):
        self.ChannelGain = self._ChannelGain_Calculated(self.sigma_data)
        state = np.concatenate((np.array(self.ChannelGain).reshape(1, -1), np.array(self.U_location).reshape(1, -1),
                                np.array(self.User_trajectory).reshape(1, -1)), axis=1)
        return state

    def _decomposeState(self, state):
        H = state[0: self.N_User]
        U_location = state[self.N_User: 2 * self.N_User + 2]
        User_trajectory = state[self.N_User + 2: 2 * self.N_User + 4]
        return [
            np.array(H), np.array(U_location), np.array(User_trajectory)
        ]

