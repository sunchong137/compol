import numpy as np
from compol import ft_civecs, hubbard 
import scipy

def test_compol_site():
    norb = 6
    nelec = (3, 3)
    T = 10
    na = 20
    nb = 20
    len_ci = na*nb
    H =  np.random.rand(len_ci, len_ci)
    H += H.T
    energies, cis = np.linalg.eigh(H)
    z = ft_civecs.ftcompol_fci_site(norb, nelec, T, energies, cis) 

    print(np.abs(z))


def test_solver():
    nsite = 8
    nelec = nsite
    len_ci = int(scipy.special.comb(nsite, nelec//2)**2)
    U = 0
    pbc = True
    h1e, eri = hubbard.hubham_1d(nsite, U, pbc)
    e, v = ft_civecs.ftfci_canonical(h1e, eri, nsite, nelec, npoint=len_ci)
    print(len(e))
    print(len_ci)


test_solver()
