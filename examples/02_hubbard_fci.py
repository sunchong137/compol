'''
Evaluate the complex polarization under FCI.
'''
import numpy as np 
from compol import hubbard, slater_uhf, helpers
from pyscf import ao2mo, fci

norb = 10
U = 4
spin = 1 
nelec = norb # half-filling
nocc = nelec // 2
x0 = 0
na = nelec // 2
nb = nelec - na

# first do a mean-field calculation
mymf = hubbard.hubbard_mf(norb, U, spin=spin, nelec=nelec, pbc=True)
rdm1 = mymf.make_rdm1()
mo_coeff = mymf.mo_coeff

h1e, eri = hubbard.hubham_1d(norb, U, pbc=True)
h1_mo, h2_mo = helpers.rotate_ham(mymf)

# FCI
cisolver = fci.direct_uhf.FCI()
e_fci, civec = cisolver.kernel(h1_mo, h2_mo, norb, (na, nb))
print(e_fci)

# compare to pyscf 
n = norb
h1 = np.zeros((n,n))
for i in range(n-1):
    h1[i,i+1] = h1[i+1,i] = -1.0
h1[n-1,0] = h1[0,n-1] = -1.0  # PBC
eri = np.zeros((n,n,n,n))
for i in range(n):
    eri[i,i,i,i] = U
myci = fci.direct_spin0
e, c = myci.kernel(h1, eri, n, n)

print(e - e_fci)