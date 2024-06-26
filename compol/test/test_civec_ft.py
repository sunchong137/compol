import numpy as np
from compol import ft_civecs, ft_civecs_spinless
from compol.hamiltonians import hubbard
import scipy
import time

def test_compol_site():
    norb = 6
    nelec = (3, 3)
    T = 10
    na = 20
    nb = 20
    len_ci = na*nb
    np.random.seed(0)
    H =  np.random.rand(len_ci, len_ci)
    H += H.T
    energies, cis = np.linalg.eigh(H)
    z = ft_civecs.ftcompol_fci_site(norb, nelec, T, energies, cis) 

    print(np.abs(z))


def test_compol_site_spinless():
    norb = 6
    nelec = 3
    T = 0.2
    na = 20
    nb = 1
    len_ci = na*nb
    np.random.seed(0)
    H =  np.random.rand(len_ci, len_ci)
    H += H.T
    energies, cis = np.linalg.eigh(H)
    z = ft_civecs_spinless.ftcompol_fci_site(norb, nelec, T, energies, cis) 

    print(np.abs(z))

test_compol_site_spinless()

def test_hmat():
    nsite = 6
    nelec = nsite
    len_ci = int(scipy.special.comb(nsite, nelec//2)**2)
    U = 0
    pbc = True
    h1e, eri = hubbard.hubham_1d(nsite, U, pbc)
    t1 = time.time()
    h = ft_civecs.fci_ham_pspace(h1e, eri, nsite, nelec, max_np=1e4)
    t2 = time.time()
    h2 = ft_civecs.fci_ham_direct(h1e, eri, nsite, nelec)
    t3 = time.time()
    print(t2-t1, t3-t2)
    assert np.allclose(h, h2)


# test_hmat()
