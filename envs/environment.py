import math

import numpy as np
from utils.setting_setup import *
import scipy

from envs.env_utils import *
from envs.env_agent_utils import *
from envs.function_utils import *

class base_env(rsma_utils, env_agent_utils):
    def __init__(self, args):
        super().__init__(args)
        self.max_step = args.max_step
        self.max_episode = args.max_episode
        self.trial = args.trial
        # Base station initialization
        self.BS_x = 0
        self.BS_y = 0
        self.BS_R_Range = args.radius_out            # Radius of cell-out
        self.BS_R_min = args.radius_in              # Radius of cell-in

        """ ========================================= """
        """ =============     Network    ============ """
        """ ========================================= """
        self.user_num = args.user_num           # Number of users
        self.antenna_num = args.antenna_num     # Number of antenna setting
        self.bandwidth = args.bandwidth         # Bandwidth in (MHz)
        self.NFdB = args.nfdb                   # From 280 GHz ~ 330 GHz

        self.PTdB = np.arange(0,20,2.5)
        self.PTdB1 = np.arange(min(self.PTdB),max(self.PTdB),1)

        self.sig2dBm = -174 + 10*np.log10(self.bandwidth) + self.NFdB   # Noise power (dBm)
        self.carrier_freq = args.carrier_freq   # Carrier Frequency (GHz) - 300 GHz

        self.lamda = 3*1e8/self.carrier_freq    # Signal wavelength
        self.kabs = args.kabs                   # Absorption Loss Coefficient Measured at 300 Ghz

        self.G_Tx = dB2pow(args.tx_db)        # Antenna Gain of Tx
        self.G_Rx = dB2pow(args.rx_db)        # Antenna Gain of Rx

        self.beta_c = args.alloc_common         # Power allocation - common packet
        self.beta_k = (1-self.beta_c)\
                      /self.user_num            # Power allocation - private packet

        self.rate_c = args.rate_common          # Rate common
        self.rate_k = args.rate_private         # Rate private

        self.g_c_threshold = \
            np.power(2,self.rate_c)               # Threshold common
        self.g_k_threshold = \
            np.power(2, self.rate_k)              # Threshold private

        self.psik = args.imperfect_sic          # Imperfect SIC coefficient


        """ ========================================= """
        """ ========== Channel Realization ========== """
        """ ========================================= """
        # Channel Realization
        self.trial = 1e4
        self.cdf_sim_c = []
        self.cdf_sim_k = []

        """ ========================================= """
        """ ===== Function-based Initialization ===== """
        """ ========================================= """
        self.BS_location = np.expand_dims(self._location_BS_Generator(), axis=0)    # Random BS location
        self.U_location = self._location_CU_Generator()                             # Random user-k location
        self.User_trajectory = self._trajectory_U_Generator()
        self.distance_CU_BS = self._distance_Calculated(self.U_location, self.BS_location)

        self.ChannelGain = self._ChannelGain_Calculated()

        """ ============================ """
        """     Environment Settings     """
        """ ============================ """
        self.rewardMatrix = np.array([])

        self.cdf_sim_c = []
        self.cdf_sim_k = []

    def step(self,step):
        """     Environment change      """
        self.User_trajectory = np.expand_dims(self._trajectory_U_Generator(), axis=0)
        self.U_location = self.User_trajectory + self.U_location
        self.distance_CU_BS = self._distance_Calculated(self.U_location, self.BS_location)

        idx_ptdb = 0
        bar_gamma = dB2pow(self.PTdB(idx_ptdb))
        count_sim_c = np.zeros((1,self.user_num))
        count_sim_k = np.zeros((1,self.user_num))

        for user in range(self.user_num):
            d_SUk = self.distance_CU_BS[user]
            # LSF * MAP part of channel hk
            lk = np.exp(-np.mean(d_SUk*self.kabs))/\
                 np.power(d_SUk,2)*np.power(self.lamda/4*np.pi(),2)\
                 *self.G_Tx*self.G_Rx

            """      Variable Initialization      """
            self.m_k = 3            # 1->4
            self.omega_k = 0.5      # 1

            # Nagakami Channel
            self.G_nagakami = np.linspace(scipy.stats.nakagami(nu=self.m_k))   # integer
            # Precoding weights
            self.W_precoding = self.G_nagakami*np.linalg.inv(T_conjugate(self.G_nagakami)*self.G_nagakami)
            """     P = [p1,p2,...,pK]      """
            self.P_precoding = self.W_precoding*np.diag(np.linalg.norm(self.G_nagakami))
            # Generate precoding weights for private message (Trial*L*1)
            self.P_k = self.P_precoding[:,user]
            # Generate precoding weight for common message (Trial*L*1)
            self.P_c = np.concatenate((2,self.P_precoding),axis=0)*T_conjugate(np.ones((1,self.user_num)))
            # Channel of user k (Trial,Antenna,1)
            self.G_k = self.G_nagakami[:,user]
            # Expect to channel norm - common |gk^h*pc|^2
            self.gkhpc = np.power(np.abs(T_conjugate(self.G_k)*self.P_c),2)
            # Expect to channel norm - private |gk^h*pk|^2
            self.gkhpk = np.power(np.abs(T_conjugate(self.G_k)*self.P_k),2)
            # Channel of other user j (L*(K-1))
            G_j = self.G_nagakami   # temporary channel
            G_j[:,user] = []           # remove channel of user k

            # Channel interference vector of other user |gj^h*pk|^2
            g_j = np.power(np.abs(T_conjugate(G_j)*self.P_k),2)
            print(self.G_k)

        next_state =
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


        return state_next

    def close(self):
        pass


if __name__ == '__main__':
    args = get_arguments()
    env = base_env(args)
