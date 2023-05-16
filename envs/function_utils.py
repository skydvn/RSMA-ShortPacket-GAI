import numpy as np


def T_conjugate(input):
    return np.tranpose(np.conjugate(input))

def dB2pow(db_inp):
    return 0.001*np.power(10,db_inp/10)

def matrix_nakagami(shape=3,scale=1,L=8,K=2):
    # Generate gamma-distributed random numbers
    gamma_samples = np.random.gamma(shape, scale=1, size=(L, K))

    # Apply transformation to obtain Nakagami distribution
    nakagami_samples = np.sqrt(gamma_samples)
    return nakagami_samples



#     # SINR at user k
#     self.gamma_kc = self.beta_c*bar_gamma*lk*self.gkhpc/\
#                     (self.beta_k*bar_gamma*lk*self.gkhpk +
#                      np.sum((self.user_num-1)*self.beta_k*bar_gamma*lk*g_j)+1)
#     self.gamma_kp = self.beta_k*bar_gamma*lk*self.gkhpk/\
#                     (np.sum((self.user_num-1)*self.beta_k*bar_gamma*lk*g_j)+
#                      self.beta_c*bar_gamma*lk*self.gkhpc*self.psik+1)
#
# # cdf_sim_c & cdf_sim_k are the list
# self.cdf_sim_c.append(1-count_sim_c[self.user_num]+1)
# self.cdf_sim_k.append(1-count_sim_k[self.user_num]+1)