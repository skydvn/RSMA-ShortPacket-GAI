import numpy as np


def T_conjugate(input):
    return np.tranpose(np.conjugate(input))

def dB2pow(db_inp):
    return 0.001*np.pow(10,db_inp/10)

