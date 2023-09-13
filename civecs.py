'''
Evaluate the complex polarization given a FCI solution.
With FCI, one should always use RHF since UHF and RHF give the same answer.
'''
import numpy as np
from pyscf.fci import direct_uhf as fcisolver
from pyscf.fci import cistring 
from pyscf.lib import numpy_helper
import slater_site

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


def compol_fci_site(L, ci, ci_strs, x0=0.0):
    '''
    Only for RHF.
    In the site basis, the determinants are eigenvalues of Z, so we only need to evaluate
    < phi_i |Z| phi_i>, and the others are non-zero.
    Caution: if the "HF" solution is too far away from the true solution,
             then FCI might not converge.
    Args:
        L: number of orbitals
        civec: FCI coefficients
    Returns
        float
    '''
    # define complex polarization
    pos = np.arange(L) + x0 
    Z = np.exp(2.j * Pi * pos / L)
    Z = np.diag(Z)

    # define site basis orbitals. 
    lmo = np.eye(L)
    rmo = np.dot(Z, lmo)

    # choose the MOs
    Z = 0
    k = 0
    len_u, len_d = ci.shape
    for u in range(len_u):
        for d in range(len_d):
            s_u = ci_strs[u]
            s_d = ci_strs[d]
            lu = slater_site.gen_det(lmo, s_u)
            ru = slater_site.gen_det(rmo, s_u)
            ld = slater_site.gen_det(lmo, s_d)
            rd = slater_site.gen_det(rmo, s_d)
            z_u = slater_site.ovlp_det(lu, ru)
            z_d = slater_site.ovlp_det(ld, rd)
            Z += z_u * z_d * (ci[u, d])**2

    return np.linalg.norm(Z)

    

def compol_fci_prod(ci, norb, nelec, x0=.0):
    '''
    Compute complex polarization given a ci vector using the following formula:
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
    NOTE: the result is off...
    '''
    ci_vec = ci + 0.j*ci # change dtype
    ci_vec = ci_vec.flatten()
    new_vec = np.copy(ci_vec)
    f0 = np.zeros((norb, norb), dtype=np.complex128)

    # spin up
    for site in range(norb):
        f1e = np.zeros((norb, norb), dtype=np.complex128)
        f1e[site, site] = np.exp(2.j * np.pi * (site+x0) / norb) - 1
        f1e_all = np.array([f1e, f0])
        delt = fcisolver.contract_1e(f1e_all, new_vec, norb, nelec)
        new_vec = np.copy(new_vec + delt)
        print(np.linalg.norm(delt))

        f1e_all = np.array([f0, f1e])
        new_vec = new_vec + fcisolver.contract_1e(f1e_all, new_vec, norb, nelec)

        #print(site, np.linalg.norm(new_vec))

    Z = np.dot(ci_vec, new_vec)
    return Z


def compol_ci_full(ci, norb, nelec, mo_coeff, x0=0.0):
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
    len_ci = ci.shape[-1]
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

    z_mo = slater_site.z_sdet(norb, mo_coeff, x0=x0)
    z_val = 0.j

    if ndim == 2: # RHF
        neleca = ne[0]
        assert np.allclose(neleca*2, ne[0] + ne[1])

        ci_strs = gen_cistr(norb, neleca)
        for up_r in range(len_ci):
            for up_l in range(len_ci):
                occ_upl = ci_strs[up_l]
                occ_upr = ci_strs[up_r]
                ket = slater_site.gen_det(z_mo, occ_upr)
                bra = slater_site.gen_det(mo_coeff, occ_upl)
                coeff = ci[up_l, up_l].conj() * ci[up_r, up_r]
                z_val += slater_site.ovlp_det(bra, ket) * coeff 

    else:
        ci_strs_up = gen_cistr(norb, ne[0])
        ci_strs_dn = gen_cistr(norb, ne[1])
        for up_r in range(len_ci):
            for dn_r in range(len_ci):
                for up_l in range(len_ci):
                    for dn_l in range(len_ci):
                        # generate the ci strings
                        occ_upl = ci_strs_up[up_l]
                        occ_dnl = ci_strs_dn[dn_l]
                        occ_upr = ci_strs_up[up_r]
                        occ_dnr = ci_strs_dn[dn_r]
                        # generate the determinants
                        ket = slater_site.gen_det(z_mo, [occ_upr, occ_dnr])
                        bra = slater_site.gen_det(mo_coeff, [occ_upl, occ_dnl])
                        coeff = ci[up_l, dn_l].conj() * ci[up_r, dn_r]
                        z_val += slater_site.ovlp_det(bra, ket) * coeff 
                                                  
    return z_val