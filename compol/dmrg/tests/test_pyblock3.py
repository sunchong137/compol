from compol.dmrg import hubbard_pyblock3 
import numpy as np

def test_hubbard1d():
    nsite = 6
    U = 4
    E, mps, mpo = hubbard_pyblock3.hubbard1d_dmrg(nsite, U)
    E_n = np.dot(mps, mpo@mps)
    assert np.allclose(E, E_n)
    
