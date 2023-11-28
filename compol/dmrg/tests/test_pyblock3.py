from compol.dmrg import hubbard_pyblock3 
import numpy as np

def test_hubbard1d():
    nsite = 6
    U = 4
    E, mps, mpo = hubbard_pyblock3.hubbard1d_dmrg(nsite, U)
    E_n = np.dot(mps, mpo@mps)
    assert np.allclose(E, E_n)
    
def test_compol_prod():
    nsite = 6
    U = 0
    E, mps, mpo = hubbard_pyblock3.hubbard1d_dmrg(nsite, U)
    Z = hubbard_pyblock3.compol_prod(mps, nsite, nsite) 
    print(Z)

test_compol_prod()