import numpy as np
import sys
sys.path.append("../")
import civecs, hubbard
from pyscf import fci
from scipy.special import comb
import time 


def test_compol_fci():
    n = 10
    U = 0
    h1 = np.zeros((n,n))
    for i in range(n-1):
        h1[i,i+1] = h1[i+1,i] = -1.0
    h1[n-1,0] = h1[0,n-1] = -1.0  # PBC
    eri = np.zeros((n,n,n,n))
    for i in range(n):
        eri[i,i,i,i] = U
    myci = fci.direct_spin1
    e, c = myci.kernel(h1, eri, n, n)
    z = civecs.compol_fci_prod(c, n, n, x0=-n/2)
    print(z)
    
    # compare to UHF

def test_gen_strs():
    norb = 4
    nelec = 2
    t1 = time.time()
    cistrs = civecs.gen_cistr(norb, nelec)
    t2 = time.time()
    print(t2-t1)
    ref = np.array([
        [1,1,0,0],
        [1,0,1,0],
        [0,1,1,0],
        [1,0,0,1],
        [0,1,0,1],
        [0,0,1,1]
    ])
    # print(cistrs)
    assert np.allclose(cistrs, ref)
# test_gen_strs()

def test_z_ci():

    #RHF 
    norb = 6
    nelec = 4
    len_ci = int(comb(norb, nelec))
    H, _ = hubbard.hamilt_hubbard(norb, U=0) 
    _, mo = np.linalg.eigh(H)
    ci = np.random.rand(len_ci, len_ci)
    ci /= np.linalg.norm(ci)
    z = civecs.compol_ci_full(ci, norb, nelec, mo)
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
    z = civecs.compol_ci_full(ci, norb, nelec, mo)
    assert np.allclose(np.abs(z), 0)


def test_compol_fci_funcs():
    #RHF 
    norb = 6
    nelec = 4
    len_ci = int(comb(norb, nelec))
    H, _ = hubbard.hamilt_hubbard(norb, U=0) 
    # _, mo = np.linalg.eigh(H) 
    mo = np.eye(norb)
    ci = np.random.rand(len_ci, len_ci)
    ci /= np.linalg.norm(ci)
    ci_strs = civecs.gen_cistr(norb, nelec//2)
    z = civecs.compol_fci_site(norb, ci, ci_strs, x0=0.0)
    z2 = civecs.compol_ci_full(ci, norb, nelec, mo, x0=0.0)
    print(z)
    print(z2)
    # assert np.abs(z) <= 1

test_compol_fci_funcs()