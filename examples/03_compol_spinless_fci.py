'''
Evaluate the complex polarization under FCI.
'''
import numpy as np 
from compol import hubbard, civecs_spinless
from pyscf import fci

norb = 16
U = 1
spin = 1 
nelec = int(norb//2+1e-6)

h1e, eri = hubbard.hubham_spinless_1d(norb, U, pbc=True)
h1_0 = np.zeros_like(h1e)
h2_0 = np.zeros_like(eri)


cisolver = fci.direct_uhf
h1_uhf = (h1e, h1_0)
h2_uhf = (eri, h2_0, h2_0)
e, c = cisolver.kernel(h1_uhf, h2_uhf, norb, (nelec, 0))

x0 = 0
z = civecs_spinless.compol_fci_site(c, norb, nelec, x0=x0)
print(z)