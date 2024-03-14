import numpy as np 
from compol import ft_civecs_spinless
from compol.hamiltonians import disorder_ham
from compol.solvers import exact_diag

nsite = 10
V = 1
tprime = 0
W = 6
T = 0.5
pbc = True
nelec = (nsite // 2, 0)
obj = disorder_ham.spinless1d(nsite, V, W, tprime, pbc, 'box')
mf = obj.run_scf()
h1e = mf.get_hcore()
h2e = mf._eri
nelec = mf.nelec
h1e_uhf = np.array([h1e, h1e*0])
h2e_uhf = np.array([h2e, h2e*0, h2e*0])
# hfci = exact_diag.fci_ham_pspace(h1e_uhf, h2e_uhf, nsite, nelec)
hfci = exact_diag.fci_ham_pspace(h1e_uhf, h2e_uhf, nsite, nelec)

energies, cis = np.linalg.eigh(hfci)

z = ft_civecs_spinless.ftcompol_fci_site(nsite, nelec, T, energies, cis, x0=0.0, ttol=1e-2, return_phase=False)
print(z)
