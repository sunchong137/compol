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
from scipy import special
# from pyscf.fci import direct_uhf as fcisolver
from pyscf.fci import fci_slow as fcisolver
from pyscf.fci import direct_spin1
from pyscf.fci import cistring 
from pyscf.lib import numpy_helper
from compol import slater, civecs

Pi = np.pi

def gen_cistr(norb, nelec):
    return civecs.gen_cistr(norb, nelec)


# def z_site_ftfci(norb, nelec, T, civecs, energies, )
def ftcompol_fci_site(norb, nelec, T, energies, cis, x0=0.0, ttol=1e-2, return_phase=False):
    '''
    Finite temperature complex polarization. 
    In the site basis, the determinants are eigenvalues of Z, so we only need to evaluate
    < phi_i |Z| phi_i>, and the others are zero.
    Args:
        norb: int, number of orbitals
        nelec: int, number of electrons
        T: float, temperature.
        energies: 1D array, energy spectrum from FCI.
        cis: 2D array, the columns are each CI eigenstate in the order of energy.
    Kwargs:
        x0: origin.
        ttol: minimum non-zero temperature.
    Returns
        float.
    '''
    if T < ttol:
        return civecs.compol_fci_site(cis[:, 0][:, None], norb, nelec, x0=x0)
    
    # define complex polarization
    zmat = slater.gen_zmat_site(norb, x0)

    # define site basis orbitals. 
    bra_mo = np.eye(norb)
    ket_mo = np.dot(zmat, bra_mo)

    bra_mo = np.array([bra_mo, bra_mo])
    ket_mo = np.array([ket_mo, ket_mo])

    try:
        neleca = nelec[0] # nelec as tuple
        ne = nelec
    except:
        neleca = nelec // 2
        nelecb = nelec - neleca
        ne = [neleca, nelecb]

    ci_strs_up = gen_cistr(norb, ne[0])
    ci_strs_dn = gen_cistr(norb, ne[1])
    
    len_u = len(ci_strs_up)
    len_d = len(ci_strs_dn)
    # choose the MOs
    Z_vals = np.zeros((len_u, len_d), dtype=complex)
    for up in range(len_u):
        for dn in range(len_d):
            occ_u = ci_strs_up[up]
            occ_d = ci_strs_dn[dn]
            
            bra = slater.gen_det(bra_mo, [occ_u, occ_d])
            ket = slater.gen_det(ket_mo, [occ_u, occ_d])
            Z_vals[up, dn] = slater.ovlp_det(bra, ket)

    Z_vals = Z_vals.ravel()
    # canonical ensemble
    weights = np.exp(-energies/T)
    top = Z_vals.T @ cis**2 @ weights 
    bot = np.sum(weights)
    Z = top/bot
    z_norm = np.linalg.norm(Z) 
    if return_phase:
        z_phase = np.angle(Z) 
        return z_norm, z_phase
    else:
        return z_norm


def ftfci_canonical(h1e, eri, norb, nelec):
    '''
    Finite temperature FCI under canonical ensemble.
    '''
    # get the Hamiltonian matrix
    H_fci = fci_ham_pspace(h1e, eri, norb, nelec)
    energies, civecs = np.linalg.eigh(H_fci) 
    return energies, civecs

def fci_ham_pspace(h1e, eri, norb, nelec, max_np=1e10):
    '''
    Construct the full Hamiltonian using the pspace method.
    maximum: 8 orbitals.
    '''
    # get the Hamiltonian matrix
    try:
        neleca, nelecb = nelec 
    except:
        neleca = nelec//2
        nelecb = nelec - neleca 

    num_np = int(special.comb(norb, neleca) * special.comb(norb, nelecb) + 1e-5)
    if num_np > max_np:
        print(f"Warning: Exceeded memeory. P space is truncated to {max_np} points.")
        num_np = max_np    
    H_fci = direct_spin1.pspace(h1e, eri, norb, nelec, np=num_np)[1]
    return H_fci 

def fci_ham_direct(h1e, eri, norb, nelec):
    '''
    Construct the FCI Hamiltonian directly.
    maximum: 8 orbitals.
    NOTE: much slower.
    '''
    try:
        neleca, nelecb = nelec
    except:
        neleca = nelec//2
        nelecb = nelec - neleca 
    nsa = int(special.comb(norb, neleca) + 1e-5)
    nsb = int(special.comb(norb, nelecb) + 1e-5) 
    len_ci = nsa * nsb

    base_mat = np.eye(len_ci)
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5)

    def hop(c):
        hc = direct_spin1.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)

    hmat = np.zeros_like(base_mat)
    for i in range(len_ci):
        hmat[:, i] = hop(base_mat[i])

    return hmat


