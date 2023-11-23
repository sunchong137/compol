from pyscf.fci import direct_uhf 
from compol import helpers
import numpy as np

def test_contract_1e_uhf():
    norb = 8
    nelec = (4, 4)
    na = 70
    nb = 70
    ha = np.random.rand(norb, norb)
    ha += ha.T 
    hb = np.random.rand(norb, norb)
    hb += hb.T 
    h1e = (ha, hb)
    ci = np.random.rand(na, nb)
    ci /= np.linalg.norm(ci)

    ci1 = helpers.contract_1e_uhf(h1e, ci, norb, nelec)
    ci2 = direct_uhf.contract_1e(h1e, ci, norb, nelec)
    ci3 = helpers.contract_1e_onespin(h1e[0], ci, norb, nelec, "a")
    ci3 += helpers.contract_1e_onespin(h1e[1], ci, norb, nelec, "b")
    assert np.allclose(ci1, ci2)
    assert np.allclose(ci1, ci3)

test_contract_1e_uhf()
    
