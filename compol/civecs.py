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
from compol import slater_uhf

Pi = np.pi

def gen_cistr(norb, nelec):
    '''
    Generate all possible string representations of the 
    occupations for a set of spin orbitals.
    Returns:
        2D array of size (norb choose nelec, norb) 
        A list of strings representing the occupation.
    Examples:
    >>> gen_cistr(4, 2)
        [[1 1 0 0]
         [1 0 1 0]
         [0 1 1 0]
         [1 0 0 1]
         [0 1 0 1]
         [0 0 1 1]]
    '''

    orb_list = np.arange(norb)
    cistrs = cistring.make_strings(orb_list, nelec)
    bin_strs = []
    for s in cistrs:
        bin_form = numpy_helper.base_repr_int(s, 2, norb)[::-1]
        # # orbital starts from the right to the left.
        bin_strs.append(bin_form)
    bin_strs = np.array(bin_strs)
    return bin_strs


def compol_fci_site(ci, L, nelec, x0=0.0):
    '''
    Only for RHF. NOTE: check this.
    In the site basis, the determinants are eigenvalues of Z, so we only need to evaluate
    < phi_i |Z| phi_i>, and the others are zero.
    Caution: if the "HF" solution is too far away from the true solution,
             then FCI might not converge.
    Args:
        L: number of orbitals
        civec: FCI coefficients
    Returns
        float
    '''
    # define complex polarization
    Z = slater_uhf.gen_zmat_site(L, x0)

    # define site basis orbitals. 
    bra_mo = np.eye(L)
    ket_mo = np.dot(Z, bra_mo)

    bra_mo = np.array([bra_mo, bra_mo])
    ket_mo = np.array([ket_mo, ket_mo])

    try:
        neleca = nelec[0] # nelec as tuple
        ne = nelec
    except:
        neleca = nelec // 2
        nelecb = nelec - neleca
        ne = [neleca, nelecb]

    ci_strs_up = gen_cistr(L, ne[0])
    ci_strs_dn = gen_cistr(L, ne[1])

    # choose the MOs
    Z = 0.j
    len_u, len_d = ci.shape
    for up in range(len_u):
        for dn in range(len_d):
            occ_u = ci_strs_up[up]
            occ_d = ci_strs_dn[dn]
            
            bra = slater_uhf.gen_det(bra_mo, [occ_u, occ_d])
            ket = slater_uhf.gen_det(ket_mo, [occ_u, occ_d])
            _z = slater_uhf.ovlp_det(bra, ket)

            coeff = ci[up, dn]*ci[up, dn].conj()
            # print(up, dn, up, dn, _z, coeff)
            Z += _z * coeff

    return np.linalg.norm(Z)


def compol_fci_prod(ci, norb, nelec, x0=0.):
    '''
    Compute complex polarization given a ci vector using the following formula:
    ...math:
    Z = exp(i 2pi/L X) = \prod (I + (exp(i 2pi/L a) - 1)N_a)
    where a is the coordinate of the orbital, and N_a is the number operator on a.

    Note: ci should be in the site basis.
    Args:
        ci: 1d array, the coefficients of the fci ground state.
        norb: int, number of orbitals.
        nelec: int or tuple, number of electrons
    Kwargs:
        x0: float, origin
    Returns:
        complex number.
    NOTE: This does't work because pyscf fci does not take complex numbers.
    '''
    ci_vec = ci.astype(complex)
    ci_vec = np.copy(ci)
    # ci_vec = ci_vec.ravel()
    new_vec = np.copy(ci_vec)
    f0 = np.zeros((norb, norb))
    # f0 = np.eye(norb, dtype=complex)
    # ne_tot = nelec[0] + nelec[1]
    # f0 /= ne_tot # equal to identity
    # f0 /= nelec[0]
    # f0_u = f0 / ne_tot
    # f0_d = f0 / ne_tot
    # na, nb = nelec
    # eri = np.zeros((3, norb, norb, norb, norb))

    for site in range(norb):
        f1e = np.zeros((norb, norb))
        f1e[site, site] = 1.0
        f1e_up = np.array([f0, f1e])
        coeff = np.exp(2.j*Pi*(site-x0)/norb)-1.0
        coeff = 1
        # h2e = fcisolver.absorb_h1e(f1e_up, eri, norb, nelec, 1)
        # spin up
        
        # delt = fcisolver.contract_2e(h2e, new_vec, norb, nelec) * coeff
        delt = fcisolver.contract_1e(f1e_up, new_vec, norb, nelec)
        # print(delt)
        # new_vec = np.copy(new_vec + delt)
        # # print(np.linalg.norm(new_vec))

        # f1e_dn = np.array([f0, f1e])
        # delt = fcisolver.contract_1e(f1e_dn, new_vec, norb, nelec) * coeff
        # new_vec = np.copy(new_vec + delt)
    Z = np.dot(ci_vec.conj(), new_vec) / np.linalg.norm(new_vec)
    
    return np.linalg.norm(Z)


def compol_fci_prod_bkup(ci, norb, nelec, x0=0.):
    '''
    Compute complex polarization given a ci vector using the following formula:
    ...math:
    Z = exp(i 2pi/L X) = \prod (I + (exp(i 2pi/L a) - 1)N_a)
    where a is the coordinate of the orbital, and N_a is the number operator on a.

    Note: ci should be in the site basis.
    Args:
        ci: 1d array, the coefficients of the fci ground state.
        norb: int, number of orbitals.
        nelec: int or tuple, number of electrons
    Kwargs:
        x0: float, origin
    Returns:
        complex number.
    NOTE: Doesn't work...
    '''
    ci_vec = ci.astype(complex)
    ci_vec = np.copy(ci)
    # ci_vec = ci_vec.ravel()
    new_vec = np.copy(ci_vec)
    f0 = np.zeros((norb, norb))
    # f0 = np.eye(norb, dtype=complex)
    # ne_tot = nelec[0] + nelec[1]
    # f0 /= ne_tot # equal to identity
    # f0 /= nelec[0]
    # f0_u = f0 / ne_tot
    # f0_d = f0 / ne_tot
    # na, nb = nelec
    eri = np.zeros((3, norb, norb, norb, norb))

    for site in range(norb):
        f1e = np.zeros((norb, norb))
        f1e[site, site] = 1.0
        f1e_up = np.array([f1e, f0])
        coeff = np.exp(2.j*Pi*(site-x0)/norb)-1.0
        coeff = 1
        h2e = fcisolver.absorb_h1e(f1e_up, eri, norb, nelec, 1)
        # spin up
        
        delt = fcisolver.contract_2e(h2e, new_vec, norb, nelec) * coeff
        print(delt)
        # new_vec = np.copy(new_vec + delt)
        # # print(np.linalg.norm(new_vec))

        # f1e_dn = np.array([f0, f1e])
        # delt = fcisolver.contract_1e(f1e_dn, new_vec, norb, nelec) * coeff
        # new_vec = np.copy(new_vec + delt)
    exit()
    Z = np.dot(ci_vec.conj(), new_vec) / np.linalg.norm(new_vec)
    
    return np.linalg.norm(Z)


def compol_fci_full(ci, norb, nelec, mo_coeff, x0=0.0):
    '''
    Evaluate the complex polarization with respect
    to a CI vector. 
    Args:
        ci: 2D numpy array, output from FCI calculations.
            The row correspond to the ci strings for up spin, and
            the columns correspond to the ci strings for down spin,
        norb: int, number of orbitals.
        nelec: int, number of electrons.
        mo_coeff: Numpy array of size (spin, norb, norb), 
                MO coefficients
    Kwargs:
        x0 : float, the origin.
    Returns:
        A complex number, the complex polarization.
    '''
    len_cistr_u = ci.shape[0]
    len_cistr_d = ci.shape[1]
    # \hat{Z} MO
    try:
        ndim = mo_coeff.ndim
    except: # list or tuple
        ndim = 3

    try:
        neleca = nelec[0] # nelec as tuple
        ne = nelec
    except:
        neleca = nelec // 2
        nelecb = nelec - neleca
        ne = [neleca, nelecb]

    if ndim == 2: # RHF
        mo_coeff = np.array([mo_coeff, mo_coeff])
        neleca = ne[0]
        assert np.allclose(neleca*2, ne[0] + ne[1])


    z_mo = slater_uhf.z_sdet(norb, mo_coeff, x0=x0)
    z_val = 0.j

    ci_strs_up = gen_cistr(norb, ne[0])
    ci_strs_dn = gen_cistr(norb, ne[1])
    for up_r in range(len_cistr_u):
        for dn_r in range(len_cistr_d):
            for up_l in range(len_cistr_u):
                for dn_l in range(len_cistr_d):
                    # generate the ci strings
                    occ_upl = ci_strs_up[up_l]
                    occ_dnl = ci_strs_dn[dn_l]
                    occ_upr = ci_strs_up[up_r]
                    occ_dnr = ci_strs_dn[dn_r]
                    # generate the determinants
                    ket = slater_uhf.gen_det(z_mo, [occ_upr, occ_dnr])
                    bra = slater_uhf.gen_det(mo_coeff, [occ_upl, occ_dnl])
                    coeff = ci[up_l, dn_l].conj() * ci[up_r, dn_r]
                    z = slater_uhf.ovlp_det(bra, ket)
                    # if up_r == up_l and dn_r == dn_l:
                    #     # print(ket)
                    #     # exit()
                    #     print(up_r, dn_r, up_l, dn_l, z, coeff)
                    # else:
                    #     assert np.linalg.norm(z)<1e-10
                    z_val +=  z * coeff 
                                                  
    return np.linalg.norm(z_val)