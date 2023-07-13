'''
Evaluate the complex polarization given a FCI solution.
'''
from pyscf.fci import direct_uhf as fcisolver
import numpy as np
# import slater_site

def compol_fci(ci, norb, nelec, x0=.0):
    '''
    Compute complex polarization given a ci vector. 
    ...math:
    Z = exp(i 2pi/L X) = \prod (I + (exp(i 2pi/L a) - 1)N_a)
    where a is the coordinate of the orbital, and N_a is the number operator on a.

    Args:
        ci: 1d array, the coefficients of the fci ground state.
        norb: int, number of orbitals.
        nelec: int or tuple, number of electrons
    Kwargs:
        x0: float, origin
    Returns:
        complex number.
    '''
    ci_vec = ci + 0.j*ci # change dtype
    ci_vec = ci_vec.flatten()
    new_vec = np.copy(ci_vec)

    for site in range(norb):
        f1e = np.zeros((norb, norb), dtype=np.complex128)
        f1e[site, site] = np.exp(2.j * np.pi * (site+x0) / norb) - 1
        f1e = np.array([f1e, f1e])
        new_vec = new_vec + fcisolver.contract_1e(f1e, new_vec, norb, nelec)
        new_vec /= np.linalg.norm(new_vec)
        #print(site, np.linalg.norm(new_vec))
        

    Z = np.dot(ci_vec, new_vec)
    return Z
