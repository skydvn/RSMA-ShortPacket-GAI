import numpy as np
from utils.setting_setup import *
import scipy


class env_agent_utils:
    def __init__(self):
        pass

    def _wrapState(self):
        state = np.concatenate((np.array(self.ChannelGain).reshape(1, -1)), axis=1)
        return state

    def _decomposeState(self, state):
        H = state[0: self.N_User]

        return [
            np.array(H)
        ]

