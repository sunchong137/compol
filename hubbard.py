'''
Hubbard model.
'''
import numpy as np
from pyscf import gto, scf, ao2mo, fci

def hubbard_mf(norb, U, spin=0, pbc=True):
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
    mol.nelectron = norb # Half-filling

    h1e = np.zeros((norb, norb))
    eri = np.zeros((norb,)*4)
    for i in range(norb-1):
        h1e[i, (i+1)] = -1.
        h1e[(i-1), i] = -1.
        eri[i,i,i,i] = U
    if pbc:
        assert norb%4 == 2, "PBC requires Norb = 4n+2!"
        h1e[0, -1] = -1.
        h1e[-1, 0] = -1.

    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h1e 
    mf.get_ovlp = lambda *args: np.eye(norb)
    mf._eri = ao2mo.restore(8, eri, norb)
    mol.incore_anyway = True
    # mo_coeff = mf.mo_coeff
    mf.kernel()
    #e_hf = mf.energy_elec
    return mf

def hubbard_fci(mf):
    cisolver = fci.FCI(mf)
    ci_energy = cisolver.kernel()[0]
    print(ci_energy)
