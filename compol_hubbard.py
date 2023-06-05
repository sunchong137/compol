'''
Evaluate the complex polarization for Hubbard model.
'''
import numpy as np 
import hubbard
import slater_site 

norb = 10
U = 8
spin = 0 
nelec = norb # half-filling
nocc = nelec // 2
x0 = 0.0

mf = hubbard.hubbard_mf(norb, U, spin=spin, nelec=nelec, pbc=True)
mo_coeff = mf.mo_coeff
sdet = mo_coeff[:, :nocc]
z = slater_site.det_z_det(norb, sdet, x0=x0)
print(z)

norb = 10
U = 8
spin = 1 
nelec = norb # half-filling
nocc = nelec // 2
x0 = 0.0

mf = hubbard.hubbard_mf(norb, U, spin=spin, nelec=nelec, pbc=True)
mo_coeff = mf.mo_coeff
sdet = mo_coeff[:, :, :nocc]
z = slater_site.det_z_det(norb, sdet, x0=x0)
print(z)
