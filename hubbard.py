'''
Standard and disordered Hubbard model.
'''
import numpy as np
import helpers
from pyscf import gto, scf, ao2mo, fci
import logging
# from scipy import special

def hubham_1d(nsite, U, pbc=True):
    '''
    Return 1D Hubbard model Hamiltonians (h1e and h2e).
    Unit: hopping amplitude t.
    Args:
        nsite : int, number of sites.
        U: double, Hubbard U/t value.
        pbc: boolean, if True, the system obeys periodic boundary condition.
    Returns:
        2D array: h1e
        4D array: h2e
    '''
    h1e = np.zeros((nsite, nsite))
    h2e = np.zeros((nsite,)*4)

    for i in range(nsite-1):
        h1e[i, (i+1)] = -1.
        h1e[(i+1), i] = -1.
        h2e[i,i,i,i] = U
    h2e[-1,-1,-1,-1] = U

    if pbc:
        # assert nsite%4 == 2, "PBC requires nsite = 4n+2!"
        h1e[0, -1] = -1.
        h1e[-1, 0] = -1.
    return h1e, h2e

def hamhub_2d():
    pass

def hubham_noisy_1d(nsite, U, pbc=True, max_w=1.0):
    '''
    Add diagonal noise to the original Hubbard model.
    The noise is a uniform distribution in [-max_w, max_w]
    '''
    h1e, h2e = hubham_1d(nsite, U, pbc)
    # generate uniform distribution
    noise = (np.random.rand(nsite)*2-1) * max_w
    noise = np.diag(noise)
    return h1e+noise, h2e

def hubham_spinless_1d(nsite, V, pbc=True):
    '''
    Spinless Hubbard model with nearest neighbor e-e interaction.
    
        :math: H = t \sum_i c^\dag_i c_{i+1} + V \sum_i n_i n_i+1
    '''
    h1e = np.zeros((nsite, nsite))
    h2e = np.zeros((nsite,)*4)

    for i in range(nsite-1):
        h1e[i, i+1] = -1.
        h1e[i+1, i] = -1.
        h2e[i, i, i+1, i+1] = V
        h2e[i+1, i+1, i, i] = V
    if pbc:
        h1e[0, -1] = h1e[-1, 0] = -1.
        h2e[0, 0, -1, -1] = h2e[-1, -1, 0, 0] = V 
    return h1e, h2e

def hubham_spinless_noisy_1d(nsite, V, pbc=True, max_w=1.0):
    '''
    Add noise to the spinless Hubbard model.
    '''
    h1e, h2e = hubham_spinless_1d(nsite, V, pbc)
    # generate uniform distribution
    noise = (np.random.rand(nsite)*2-1) * max_w
    noise = np.diag(noise)
    return h1e+noise, h2e

def hubbard_mf(nsite, U, spin=0, nelec=None, pbc=True, filling=1.0):
    '''
    Mean-field of Hubbard model.
    Args:
        nsite: int, number of sites
        U: positive float, Hubbard U/t
    Kwargs:
        spin: 0 (RHF) or 1 (UHF)
        nelec: int, number of electrons.
        pbc: boolean, periodic boundary condition
        filling: float, nelec/nsite. value 1 corresponds to half-filling.
    Returns:
        float: HF energy 
        numpy array of size (nspin, nsite, nsite), the MO coefficients.
    '''
    # nsite need to be 4n + 2, otherwise there is degeneracy between HOMO and LUMO
    mol = gto.M()

    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
    if abs(nelec - nsite * filling) > 1e-2:
        logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer number of electrons!".format(filling, nelec/nsite))
    
    mol.nelectron = nelec
    h1e, eri = hubham_1d(nsite, U, pbc=pbc)
    
    if spin == 0:
        mf = scf.RHF(mol)
    elif spin == 1:
        mf = scf.UHF(mol)
    else:
        raise ValueError("Spin has to be 0 or 1!")

    mf.get_hcore = lambda *args: h1e 
    mf.get_ovlp = lambda *args: np.eye(nsite)
    mf._eri = ao2mo.restore(8, eri, nsite)
    mol.incore_anyway = True

    if spin == 1:
        helpers.run_stab_mf(mf)
    else:
        mf.kernel()
    return mf

def hubbard_spinless_mf(nsite, V, nelec=None, pbc=True, filling=1.0):
    '''
    Mean-field for spinless Hubbard model.
    '''
    mol = gto.M()
    filling /= 2 # spinless 
    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
    if abs(nelec - nsite * filling) > 1e-2:
        logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer number of electrons!".format(filling, nelec/nsite))
    

    mol.nelectron = nelec
    mol.spin = nelec
    h1e, eri = hubham_spinless_1d(nsite, V, pbc=pbc)
    
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h1e 
    mf.get_ovlp = lambda *args: np.eye(nsite)
    mf._eri = ao2mo.restore(8, eri, nsite)
    mol.incore_anyway = True
    mf.kernel()

    return mf


def hubbard_fci(nsite, U, nelec=None, pbc=True, filling=1.0):
    
    h1e, eri = hubham_1d(nsite, U, pbc)

    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
    if abs(nelec - nsite * filling) > 1e-2:
        logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer number of electrons!".format(filling, nelec/nsite))
    
    # initial guess
    try:
        na, nb = nelec 
    except:
        na = nelec//2
        nb = nelec - na

    if na == nb:
        cisolver = fci.direct_spin0
    else:
        cisolver = fci.direct_spin1
        nelec = (na, nb)
    # len_a = int(special.comb(nsite, na) + 1e-10)
    # len_b = int(special.comb(nsite, nb) + 1e-10)
    # ci0 = np.random.rand(len_a, len_b)
    # ci0 = np.ones((len_a, len_b)) / np.sqrt(len_a*len_b)

    e, c = cisolver.kernel(h1e, eri, nsite, nelec)
    return e, c


def hubbard_spinless_fci(nsite, V, nelec=None, pbc=True, filling=1.0):

    h1e, eri = hubham_spinless_1d(nsite, V, pbc)
    h1_0 = np.zeros_like(h1e)
    h2_0 = np.zeros_like(eri)

    filling /= 2
    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
    if abs(nelec - nsite * filling) > 1e-2:
        logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer number of electrons!".format(filling, nelec/nsite))

    cisolver = fci.direct_uhf
    h1_uhf = (h1e, h1_0)
    h2_uhf = (eri, h2_0, h2_0)
    # h1_uhf = (h1e, h1e)
    # h2_uhf = (eri, eri, eri)
    e, c = cisolver.kernel(h1_uhf, h2_uhf, nsite, (nelec, 0))
    return e, c


def hubbard_fci_from_mf(mf):
    '''
    This one is faster because the initial guess is better.
    '''
    cisolver = fci.FCI(mf)
    ci_energy, ci = cisolver.kernel()
    
    return ci_energy, ci
