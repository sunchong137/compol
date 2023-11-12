from pyscf import gto, scf, ao2mo
import logging
import numpy as np
from compol import hubbard

nsite = 10
nelec = 5
filling =0.5 
mol = gto.M()
V = 4

mol.nelectron = nelec
mol.nao = nsite
mol.spin = 5
h1e, eri = hubbard.hubham_spinless_1d(nsite, V, pbc=True)

mf = scf.UHF(mol)
mf.get_hcore = lambda *args: h1e 
mf.get_ovlp = lambda *args: np.eye(nsite)
mf._eri = ao2mo.restore(8, eri, nsite)
mol.incore_anyway = True
mf.kernel()
mo_coeff = mf.mo_coeff
print(mo_coeff.shape)