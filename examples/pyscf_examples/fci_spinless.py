
import numpy as np
from pyscf import fci
from compol import hubbard

nsite = 10
V = 4
filling = 0.5
nelec = 5
    
h1e, eri = hubbard.hubham_spinless_1d(nsite, V, pbc=True)
h1_0 = np.zeros_like(h1e)
h2_0 = np.zeros_like(eri)

cisolver = fci.direct_uhf
h1_uhf = (h1e, h1_0)
h2_uhf = (eri, h2_0, h2_0)
# h1_uhf = (h1e, h1e)
# h2_uhf = (eri, eri, eri)
e, c = cisolver.kernel(h1_uhf, h2_uhf, nsite, (nelec, 0))
print(e)
print(c.shape)