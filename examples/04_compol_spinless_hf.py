import numpy as np 
from compol import slater_spinless
from compol.hamiltonians import disorder_ham
from compol.solvers import fci_uhf

nsite = 30
V = 0
tprime = 0
W = 10
pbc = True
nelec = (nsite // 2, 0)
obj = disorder_ham.spinless1d(nsite, V, W, tprime, pbc, 'box')
mf = obj.run_scf()
mo_coeff = mf.mo_coeff[0]
nocc = int(np.sum(mf.mo_occ[0]) + 1e-10)
sdet = mo_coeff[:, :nocc]
z = slater_spinless.det_z_det(nsite, sdet, x0=0.0)
print(z)

