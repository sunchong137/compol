import numpy as np
import sys
sys.path.append("../")
import civecs, hubbard
from scipy.special import comb

def test_gen_strs():
    norb = 4
    nelec = 2
    cistrs = civecs.gen_cistr(norb, nelec)
    ref = np.array([
        [1,1,0,0],
        [1,0,1,0],
        [0,1,1,0],
        [1,0,0,1],
        [0,1,0,1],
        [0,0,1,1]
    ])
    print(cistrs)
    assert np.allclose(cistrs, ref)

def test_z_ci():

    #RHF 
    norb = 6
    nelec = 4
    len_ci = int(comb(norb, nelec))
    H, _ = hubbard.hamilt_hubbard(norb, U=0) 
    _, mo = np.linalg.eigh(H)
    ci = np.random.rand(len_ci, len_ci)
    ci /= np.linalg.norm(ci)
    z = civecs.compol_ci(ci, norb, nelec, mo)
    assert np.abs(z) <= 1
    
    # UHF 
    norb = 6
    nelec = 6
    len_ci = int(comb(norb, nelec))
    H, _ = hubbard.hamilt_hubbard(norb, U=0) 
    _, mo = np.linalg.eigh(H)
    mo = np.array([mo, mo])
    ci = np.random.rand(len_ci, len_ci)
    ci /= np.linalg.norm(ci)
    ci = np.zeros((len_ci, len_ci))
    ci[0,0] = 1
    z = civecs.compol_ci(ci, norb, nelec, mo)
    assert np.allclose(np.abs(z), 0)


test_z_ci()