import numpy as np 
from compol import ft_civecs_spinless, helpers
from compol.hamiltonians import disorder_ham
from compol.solvers import fci_uhf, exact_diag

nsite = 10
V = 0
tprime = 0
W = 3
T = 0.2
pbc = True
nelec = (nsite // 2, 0)
obj = disorder_ham.spinless1d(nsite, V, W, tprime, pbc, 'box')
mf = obj.run_scf()
h1e = mf.get_hcore()
h2e = mf._eri
nelec = mf.nelec
mo = mf.mo_coeff
e_hf = mf.e_tot
h1e_mo, h2e_mo = helpers.ao2mo_spinless(h1e, h2e, mo)
h1e_uhf = np.array([h1e, h1e*0])
h2e_uhf = np.array([h2e, h2e*0, h2e*0])
# hfci = exact_diag.fci_ham_pspace(h1e_uhf, h2e_uhf, nsite, nelec)
hfci = exact_diag.fci_ham_pspace(h1e_uhf, h2e_uhf, nsite, nelec)

energies, cis = np.linalg.eigh(hfci)

z = ft_civecs_spinless.ftcompol_fci_site(nsite, nelec, T, energies, cis, x0=0.0, ttol=1e-2, return_phase=False)
print(z)
