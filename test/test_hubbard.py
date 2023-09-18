import numpy as np
import sys
sys.path.append("../")
import hubbard, helpers
import slater_site
from pyscf import gto, scf, ao2mo, fci
import time


def test_mf_rhf():
    norb = 6
    U = 4
    spin = 0
    mymf = hubbard.hubbard_mf(norb, U, spin=spin)
    e = mymf.energy_elec()[0]
    mo = mymf.mo_coeff 

    # pyscf example
    mymol = gto.M() 
    n = 6
    mymol.nelectron = n

    mf = scf.RHF(mymol)
    h1 = np.zeros((n,n))
    for i in range(n-1):
        h1[i,i+1] = h1[i+1,i] = -1.0
    h1[n-1,0] = h1[0,n-1] = -1.0  # PBC
    eri = np.zeros((n,n,n,n))
    for i in range(n):
        eri[i,i,i,i] = 4.0

    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(n)
    # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
    # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
    mf._eri = ao2mo.restore(8, eri, n)

    mf.kernel()

    e_hf = mf.energy_elec()[0]
    mo_coeff = mf.mo_coeff

    assert np.allclose(e, e_hf)
    assert np.allclose(mo, mo_coeff)
    
def test_uhf():
    norb = 18
    U = 8
    spin = 1
    mymf = hubbard.hubbard_mf(norb, U, spin=spin)
    e = mymf.energy_elec()[0]
    mo = mymf.mo_coeff 

    # pyscf example
    mymol = gto.M() 
    mymol.nelectron = norb

    mf = scf.UHF(mymol)
    h1 = np.zeros((norb, norb))
    for i in range(norb - 1):
        h1[i,i+1] = h1[i+1,i] = -1.0
    h1[norb-1,0] = h1[0,norb-1] = -1.0  # PBC
    eri = np.zeros((norb,)*4)
    for i in range(norb):
        eri[i,i,i,i] = U

    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(norb)
    # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
    # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
    mf._eri = ao2mo.restore(8, eri, norb)

    helpers.run_stab_mf(mf)
    e_hf = mf.energy_elec()[0]
    mo_coeff = mf.mo_coeff

    assert np.allclose(e, e_hf)

def test_fci():
    bethe = -0.573729
    norb = 6
    U = 4
    pbc = True
    spin = 0
    # t1 = time.time()
    mymf = hubbard.hubbard_mf(norb, U, spin=spin, pbc=pbc)
    e_fci, ci = hubbard.hubbard_fci_from_mf(mymf)
    # t2 = time.time()
    e_fci2, ci2 = hubbard.hubbard_fci(norb, U, pbc=pbc)
    # t3 = time.time()
    assert np.allclose(e_fci, e_fci2)
    # print(t2-t1, t3-t2)
    # exit()
    e_per_site = e_fci / norb
    assert abs(e_per_site - bethe) < 1e-1

# test_fci()