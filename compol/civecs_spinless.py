# Copyright 2023 ComPol developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Evaluate the complex polarization given a FCI solution.
With FCI, one should always use RHF since UHF and RHF give the same answer.
'''
import numpy as np
from pyscf.fci import direct_uhf as fcisolver
from pyscf.fci import cistring 
from pyscf.lib import numpy_helper
from compol import slater_spinless, civecs

Pi = np.pi

def gen_cistr(norb, nelec):
    return civecs.gen_cistr(norb, nelec)


def compol_fci_site(ci, L, nelec, x0=0.0):
    '''
    In the site basis, the determinants are eigenvalues of Z, so we only need to evaluate
    < phi_i |Z| phi_i>, and the others are zero.
    Args:
        L: number of orbitals
        civec: FCI coefficients
    Returns
        float
    '''
    # define complex polarization
    Z = slater_spinless.gen_zmat_site(L, x0)
    ci = ci.ravel()
    # define site basis orbitals. 
    bra_mo = np.eye(L)
    ket_mo = np.dot(Z, bra_mo)
    ci_strs = gen_cistr(L, nelec)
    # choose the MOs
    Z = 0.j
    len_ci = len(ci)
    for iter in range(len_ci):
        occ = ci_strs[iter]
        idx = np.nonzero(occ)[0]
        bra = bra_mo[:, idx]
        ket = ket_mo[:, idx]
        _z = slater_spinless.ovlp_det(bra, ket)
        coeff = ci[iter]*ci[iter].conj()
        Z += _z * coeff
    return np.linalg.norm(Z)


def compol_fci_prod(ci, norb, nelec, x0=0.):
    raise ValueError("Not implemented!")


def compol_fci_full(ci, norb, nelec, mo_coeff, x0=0.0):
    '''
    Evaluate the complex polarization with respect
    to a CI vector. 
    Args:
        ci: 1D numpy array, output from FCI calculations.
            The row correspond to the ci strings for up spin, and
            the columns correspond to the ci strings for down spin,
        norb: int, number of orbitals.
        nelec: int, number of electrons.
        mo_coeff: Numpy array of size (norb, norb), MO coefficients
    Kwargs:
        x0 : float, the origin.
    Returns:
        A complex number, the complex polarization.
    '''

    ci = ci.ravel()
    len_ci = len(ci)
   
    z_mo = slater_spinless.z_sdet(norb, mo_coeff, x0=x0)
    z_val = 0.j

    ci_strs = gen_cistr(norb, nelec)
    for iter_l in range(len_ci):
        for iter_r in range(len_ci):
            occ_l = ci_strs[iter_l]
            occ_r = ci_strs[iter_r]

            # generate the determinants
            ket = slater_spinless.gen_det(z_mo, occ_r)
            bra = slater_spinless.gen_det(mo_coeff, occ_l)
            coeff = ci[iter_l].conj() * ci[iter_r]
            z = slater_spinless.ovlp_det(bra, ket)
            z_val +=  z * coeff 
                                            
    return np.linalg.norm(z_val)