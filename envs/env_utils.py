import numpy as np
from utils.setting_setup import *
import scipy

class rsma_utils():
    def __init__(self, args):

        """ ========================================= """
        """ =============     Network    ============ """
        """ ========================================= """

    def _location_BS_Generator(self):
        BS_location = [self.BS_x, self.BS_y]
        return np.array(BS_location)

    # Iot initialization
    def _location_CU_Generator(self):
        userList = []
        # hUser_temp = 1.65
        for i in range(self.user_num):
            r = self.BS_R_Range * np.sqrt(np.random.rand()) + self.BS_R_min
            theta = np.random.uniform(-np.pi, np.pi)
            xUser_temp = self.BS_x + r * np.cos(theta)
            yUser_temp = self.BS_y + r * np.sin(theta)
            userList.append([xUser_temp, yUser_temp])
            U_location = np.array(userList)
            # print(U_location)
        return U_location

    def _trajectory_U_Generator(self):
        userList = []
        for i in range(self.user_num):
            theta = 0
            theta = theta + np.pi / 360
            r = np.sin(theta)
            xUser_temp = r * np.cos(2 * theta)
            yUser_temp = r * np.sin(2 * theta)
            userList.append([xUser_temp, yUser_temp])
            User_trajectory = np.array(userList)
        return User_trajectory


    def _distance_Calculated(self, A, B):
        return np.array([np.sqrt(np.sum((A - B) ** 2, axis=1))]).transpose()

    # def _distance_Calculated(self):
    #       dist = np.zeros((self.Num_BS, self.N_User))
    #       for i in range(self.Num_BS):
    #           for j in range(self.N_User):
    #               dist[i][j] = np.sqrt(np.sum(self.BS_location[i]-self.U_location[j])**2)

    #       return dist

    def _ChannelGain_Calculated(self):

        return None

    def _calculateDataRate(self, channelGain_BS_CU):

        return None

