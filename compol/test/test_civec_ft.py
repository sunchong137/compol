import numpy as np
from compol import ft_civecs 

def test_compol_site():
    norb = 6
    nelec = (3, 3)
    T = 0.1
    na = 20
    nb = 20
    len_ci = na*nb
    H =  np.random.rand(len_ci, len_ci)
    H += H.T
    energies, cis = np.linalg.eigh(H)
    z = ft_civecs.ftcompol_fci_site(norb, nelec, T, energies, cis) 
    print(z)


test_compol_site()