'''
Hubbard model.
'''
import numpy as np
from pyscf import gto, scf, ao2mo, fci

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

    h1e = np.zeros((norb, norb))
    eri = np.zeros((norb,)*4)

    for i in range(norb-1):
        h1e[i, (i+1)] = -1.
        h1e[(i+1), i] = -1.
        eri[i,i,i,i] = U
    eri[-1,-1,-1,-1] = U
    if pbc:
        assert norb%4 == 2, "PBC requires Norb = 4n+2!"
        h1e[0, -1] = -1.
        h1e[-1, 0] = -1.

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
        # because there is degeneracy in the orbitals, different init_guess will give different mo_coeffs
        # but the energy is the same.
        init_guess = mf.get_init_guess()
        for i in range(norb//2):
            init_guess[0][i*2+1, i*2+1] = 1
            init_guess[1][i*2+1, i*2+1] = 0
            init_guess[0][i*2, i*2] = 0
            init_guess[1][i*2, i*2] = 1
        mf.init_guess = init_guess

    mf.kernel()
    return mf

def hubbard_fci(mf):
    cisolver = fci.FCI(mf)
    ci_energy, ci = cisolver.kernel()
    
    return ci_energy, ci
