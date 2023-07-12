'''
Evaluate the complex polarization under FCI.
'''
import numpy as np 
import sys
sys.path.append("../")
import hubbard
import slater_site 
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

# rotate h1e 
h1e, eri = hubbard.hamilt_hubbard(norb, U, pbc=True)
h1_a = mo_coeff[0].T.conj()@h1e@mo_coeff[0]
h1_b = mo_coeff[1].T.conj()@h1e@mo_coeff[1]
h1e_n = np.array([h1_a, h1_b])

# aaaa, aabb, bbbb
Ca, Cb = mo_coeff[0], mo_coeff[1]
aaaa = (Ca,)*4
bbbb = (Cb,)*4
aabb = (Ca, Ca, Cb, Cb)

h2e_aaaa = ao2mo.incore.general(mymf._eri, aaaa, compact=False).reshape(norb, norb, norb, norb)
h2e_bbbb = ao2mo.incore.general(mymf._eri, bbbb, compact=False).reshape(norb, norb, norb, norb)
h2e_aabb = ao2mo.incore.general(mymf._eri, aabb, compact=False).reshape(norb, norb, norb, norb)
h2e_n = np.array([h2e_aaaa, h2e_aabb, h2e_bbbb])
# FCI
cisolver = fci.direct_uhf.FCI()
e_fci, civec = cisolver.kernel(h1e_n, h2e_n, norb, (na, nb))
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