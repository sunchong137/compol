'''
Hubbard model.
'''
import numpy as np
import helpers
from pyscf import gto, scf, ao2mo, fci
from scipy import special

def hubham_1d(norb, U, pbc=True):
    h1e = np.zeros((norb, norb))
    eri = np.zeros((norb,)*4)

    for i in range(norb-1):
        h1e[i, (i+1)] = -1.
        h1e[(i+1), i] = -1.
        eri[i,i,i,i] = U
    eri[-1,-1,-1,-1] = U

    if pbc:
        # assert norb%4 == 2, "PBC requires Norb = 4n+2!"
        h1e[0, -1] = -1.
        h1e[-1, 0] = -1.
    return h1e, eri

def hamhub_2d():
    pass

def hubham_noisy_1d(norb, U, pbc=True, max_w=1.0):
    '''
    Add diagonal noise to the original Hubbard model.
    The noise is a uniform distribution in [-max_w, max_w]
    '''
    h1e, eri = hubham_1d(norb, U, pbc)
    # generate uniform distribution
    noise = (np.random.rand(norb)*2-1) * max_w
    noise = np.diag(noise)
    return h1e+noise, eri

def hubham_spinless_1d():
    '''
    Spinless Hubbard model with nearest neighbor e-e interaction.
    '''
    pass

def hubbard_mf(norb, U, spin=0, nelec=None, pbc=True):
    '''
    Mean-field of Hubbard model.
    Args:
        norb: int, number of sites
        U: positive float, Hubbard U
    Kwargs:
        spin: 0 (RHF) or 1 (UHF)
        pbc: boolean, periodic boundary condition
    Returns:
        float: HF energy 
        numpy array of size (nspin, norb, norb), the MO coefficients.
    '''
    # Norb need to be 4n + 2, otherwise there is degeneracy between HOMO and LUMO
    mol = gto.M()
    if nelec is None:
        nelec = norb 
    mol.nelectron = norb

    h1e, eri = hubham_1d(norb, U, pbc=pbc)
    
    if spin == 0:
        mf = scf.RHF(mol)
    elif spin == 1:
        mf = scf.UHF(mol)
    else:
        raise ValueError("Spin has to be 0 or 1!")

    mf.get_hcore = lambda *args: h1e 
    mf.get_ovlp = lambda *args: np.eye(norb)
    mf._eri = ao2mo.restore(8, eri, norb)
    mol.incore_anyway = True

    if spin == 1:
        helpers.run_stab_mf(mf)
    else:
        mf.kernel()
    return mf

def hubbard_fci(norb, U, spin=0, nelec=None, pbc=True):
    
    h1e, eri = hubham_1d(norb, U, pbc)
    cisolver = fci.direct_spin0
    if nelec is None:
        nelec = norb
    # initial guess
    try:
        na, nb = nelec 
    except:
        na = nelec//2
        nb = nelec - na
    
    len_a = int(special.comb(norb, na) + 1e-10)
    len_b = int(special.comb(norb, nb) + 1e-10)
    ci0 = np.random.rand(len_a, len_b)
    # ci0 = np.ones((len_a, len_b)) / np.sqrt(len_a*len_b)

    e, c = cisolver.kernel(h1e, eri, norb, nelec)
    return e, c

def hubbard_fci_from_mf(mf):
    '''
    This one is faster because the initial guess is better.
    '''
    cisolver = fci.FCI(mf)
    ci_energy, ci = cisolver.kernel()
    
    return ci_energy, ci
