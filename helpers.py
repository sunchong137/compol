import numpy as np


def make_init_guess(dm0):
    '''
    Break symmetry of the UHF initial guess.
    '''
    norb = dm0[0].shape[-1]
    for i in range(norb//2): # make afm initial guess
        dm0[0][i*2+1, i*2+1] = 1.
        dm0[1][i*2+1, i*2+1] = 0.
        dm0[0][i*2, i*2] = 0.
        dm0[1][i*2, i*2] = 1.
    return dm0