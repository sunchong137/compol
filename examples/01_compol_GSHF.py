'''
Evaluate the ground state complex polarization for Hubbard model.
'''
import sys
sys.path.append("../")
import hubbard
import slater_site 

norb = 10
U = 4
spin = 1 
nelec = norb # half-filling
nocc = nelec // 2
x0 = 0

mymf = hubbard.hubbard_mf(norb, U, spin=spin, nelec=nelec, pbc=True)
rdm1 = mymf.make_rdm1()
#print(rdm1[0] - rdm1[1])
mo_coeff = mymf.mo_coeff
print(mo_coeff[0] - mo_coeff[1])
sdet = mo_coeff[:, :, :nocc]
z = slater_site.det_z_det(norb, sdet, x0=0.0)
print(z)
