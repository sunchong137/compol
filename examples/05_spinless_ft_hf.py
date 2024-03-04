import numpy as np 
from compol import ft_slater_spinless
from compol.hamiltonians import disorder_ham

T = 0.5
nsite = 10
V = 1
tprime = 0
W = 1
pbc = True
nelec = (nsite // 2, 0)
obj = disorder_ham.spinless1d(nsite, V, W, tprime, pbc, 'box')
mf = obj.run_scf(T=T)
# fock = mf.get_fock()[0]
Z = ft_slater_spinless.det_z_det_iter(nsite, mf, T, x0=0, Tmin=2e-2, return_phase=False)
Z2 = ft_slater_spinless.det_z_det(nsite, mf, T, x0=0, Tmin=2e-2, return_phase=False)

print(Z)
print(Z2)