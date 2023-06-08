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

def test_gen_det():
    mo_coeff = np.random.rand(3,3)
    occ = np.array([1,0,1])
    det = slater_site.gen_det(mo_coeff, occ)
    ref = np.zeros((3,2))
    ref[:, 0] = mo_coeff[:, 0]
    ref[:, 1] = mo_coeff[:, 2]
    assert np.allclose(det, ref)

    # UHF
    mo_up = np.random.rand(3,3)
    mo_dn = np.random.rand(3,3)

    occ_up = np.array([1,0,1])
    occ_dn = np.array([0,1,0])

    ref_u = np.zeros((3,2))
    ref_u[:, 0] = mo_up[:, 0]
    ref_u[:, 1] = mo_up[:, 2]

    ref_d = np.zeros((3, 1))
    ref_d[:, 0] = mo_dn[:, 1]
    
    ref = [ref_u, ref_d]
    det = slater_site.gen_det([mo_up, mo_dn], [occ_up, occ_dn])
    assert np.allclose(ref[0], det[0])
    assert np.allclose(ref[1], det[1])

test_gen_det()

