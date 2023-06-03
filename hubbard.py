'''
Hubbard model.
'''
import numpy as np
from pyscf import gto, scf, ao2mo, fci

# Norb need to be 4n + 2, otherwise there is degeneracy between HOMO and LUMO
norb = 6
U = 4  
mol = gto.M()
mol.nelectron = norb # Half-filling

h1e = np.zeros((norb, norb))
eri = np.zeros((norb,)*4)
for i in range(norb):
    h1e[i, (i+1)%norb] = -1.
    h1e[i, (i-1)%norb] = -1.
    eri[i,i,i,i] = U


mf = scf.UHF(mol)
mf.get_hcore = lambda *args: h1e 
mf.get_ovlp = lambda *args: np.eye(norb)
mf._eri = ao2mo.restore(8, eri, norb)
mol.incore_anyway = True


cisolver = fci.direct_spin1()
ci_energy = cisolver.kernel()[0]
print(ci_energy)
