import numpy as np 
from compol import civecs_spinless
from compol.hamiltonians import disorder_ham
from compol.solvers import fci_uhf

nsite = 14
V = 1
tprime = 0
W = 1
pbc = True
nelec = (nsite // 2, 0)
obj = disorder_ham.spinless1d(nsite, V, W, tprime, pbc, 'box')
mf = obj.run_scf()
h1e = mf.get_hcore()
h2e = mf._eri
nelec = mf.nelec
mo = mf.mo_coeff
e_hf = mf.e_tot
# h1e_mo, h2e_mo = helpers.ao2mo_spinless(h1e, h2e, mo)
# e, v = fci_uhf.kernel(h1e_mo, h2e_mo, nsite, nelec, target_e=0)
# # h1e, eri = obj.gen_ham_uhf()
h1e_uhf = np.array([h1e, h1e*0])
h2e_uhf = np.array([h2e, h2e*0, h2e*0])
# e, v = fci_uhf.kernel(h1e_uhf, h2e_uhf, nsite, nelec, target_e=0)
e, v = fci_uhf.kernel(h1e_uhf, h2e_uhf, nsite, nelec, target_e=None)

x0 = 10
# z = civecs_spinless.compol_fci_prod(v, nsite, nelec, x0=x0)
z = civecs_spinless.compol_fci_site(v, nsite, nelec, x0=x0)
print(z)