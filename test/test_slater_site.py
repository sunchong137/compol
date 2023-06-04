import sys 
import numpy as np
sys.path.append("../")
import slater_site 

Pi = np.pi

def test_rotation():
    L = 6
    nocc = 3
    x0 = 0.1
    mo = np.random.rand(2, L, nocc)
    zc = slater_site.z_sdet(L, mo, x0=x0)
    i = np.random.randint(L)
    shift = np.exp(2.j * Pi * (i+x0)/L)
    assert np.allclose(zc[1, i, :], shift*mo[1, i, :])

def test_ovlp():
    L = 6
    nocc = 3
    mo1 = np.random.rand(2, L, nocc)
    mo2 = slater_site.z_sdet(L, mo1, x0=0)
    ovlp = slater_site.ovlp_det(mo1, mo2)
    print(ovlp)

def test_get_z():
    L = 6
    nocc = 3
    mo1 = np.random.rand(2, L, nocc)
    Z = slater_site.det_z_det(L, mo1)
    print(Z)


test_get_z()
