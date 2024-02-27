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
Evaluate complex polarization based on Slater determinants at finite temperature.
1D model system with site basis only, 
'''
from compol import slater_spinless
import numpy as np
from scipy import linalg as sla
from scipy.optimize import minimize

Pi = np.pi

def gen_zmat_site(L, x0):
    return slater_spinless.gen_zmat_site(L, x0)

def gen_zmat_site_extend(L, x0):
    '''
    Generate the matrix for Z operator in the site basis for MO coeffs.
    '''
    pos = np.arange(L) + x0 
    Z = np.eye(L*2, dtype=np.complex128)
    Z[:L, :L] = np.diag(np.exp(2.j * Pi * pos / L))
    return Z 

def ovlp_det(sdet1, sdet2, ao_ovlp=None):
    return slater_spinless.ovlp_det(sdet1, sdet2, ao_ovlp=ao_ovlp)

def rdm1_ft(mf):
    '''
    Evaluate the rdm. 
    '''
    mo = mf.mo_coeff[0]
    occ = mf.mo_occ[0] 
    rdm1 = mo @ occ @ mo.T
    return rdm1

def det_z_det(L, mf, T, x0=0, Tmin=2e-1, mu=None, return_phase=False, 
              max_iter=500):
    '''
    Finite temperature form of the complex polarization.
    Args:
        L (int) : length of the site.
        fock (array) : the finite T fock operator.
        T (float) : temperature.
    Kwargs:
        x0 (float) : original
        Tmin (float) : minimum non-zero temperature.
    Returns:
        float, the modulo of the complex polarization.
    '''
    if T < Tmin:
        print("WARNING: Falling back to ground state.")
        mo_coeff = mf.mo_coeff[0]
        nocc = int(np.sum(mf.mo_occ[0]) + 1e-10)
        sdet = mo_coeff[:, :nocc]
        return slater_spinless.det_z_det(L, sdet, x0=x0, return_phase=return_phase)
    beta = 1/T
    zmat = gen_zmat_site(L, x0) 
    Imat = np.eye(L, dtype=np.complex128) 
    # get hamiltonian
    fock = mf.get_fock()[0]
    if mu is None:
        nelec = np.sum(mf.nelec) 
        mu = get_mu(fock, nelec, beta, mu0=0)
    mu_mat = np.eye(L) * mu
    hcore_beta = -beta * (fock - mu_mat)
    rho = sla.expm(hcore_beta) 
    _, e_top, _ = np.linalg.svd(zmat @ rho + Imat)
    _, e_bot, _ = np.linalg.svd(rho + Imat)
    s_top = np.prod(np.sign(e_top))
    s_bot = np.prod(np.sign(e_bot))
    sign = s_top * s_bot
    # log_z = np.sum(np.log(e_top)) - np.sum(np.log(e_bot))
    log_z = np.sum(np.log(np.abs(e_top))) - np.sum(np.log(np.abs(e_bot)))
    Z = np.exp(log_z) * sign
    # Z = np.prod(e_top/e_bot)
    # sign_t = np.prod(np.sign(e_top))
    # top = np.prod(e_top)
    # bot = np.prod(e_bot)
    
    # # top = np.linalg.det(zmat @ rho + Imat)
    # # bot = np.linalg.det(rho + Imat)
    # Z = top / bot 
    # print(bot, top)
    z_norm = np.linalg.norm(Z)
    return z_norm


def det_z_det_old(L, mf, T, x0=0, Tmin=2e-1, mu=None, return_phase=False, 
              max_iter=500):
    '''
    Finite temperature form of the complex polarization.
    Args:
        L (int) : length of the site.
        fock (array) : the finite T fock operator.
        T (float) : temperature.
    Kwargs:
        x0 (float) : original
        Tmin (float) : minimum non-zero temperature.
    Returns:
        float, the modulo of the complex polarization.
    '''
    if T < Tmin:
        print("WARNING: Falling back to ground state.")
        mo_coeff = mf.mo_coeff[0]
        nocc = int(np.sum(mf.mo_occ[0]) + 1e-10)
        sdet = mo_coeff[:, :nocc]
        return slater_spinless.det_z_det(L, sdet, x0=x0, return_phase=return_phase)
    beta = 1/T
    zmat = gen_zmat_site_extend(L, x0) 
    Imat = np.eye(L)
    # get hamiltonian
    fock = mf.get_fock()[0]
    if mu is None:
        nelec = np.sum(mf.nelec) 
        mu = get_mu(fock, nelec, beta, mu0=0)
    mu_mat = np.eye(L) * mu
    hcore = fock - mu_mat
    ew, ev = np.linalg.eigh(hcore)

    rho = np.eye(L*2, dtype=np.complex128) 
    rho[:L, :L] = sla.expm(-beta * (fock - mu_mat)) 
    # x = np.max(rho)

    C0 = np.zeros((2*L, L), dtype=np.complex128)
    C0[:L] = np.eye(L)
    C0[L:] = np.eye(L)

    # rescale C0 for stability 
    fac = 0.2
    step = 0.9
    conv = False
    for iter in range(max_iter):
        rho_c0 = rho @ C0 
        bot = np.linalg.det(C0.T @ rho_c0)
        if bot > 1e30:
            fac *= step
        elif bot < 1e-30:
            fac /= step
        else:
            conv = True
            break
        C0 *= fac 
    if conv:
        top = np.linalg.det(C0.T @ zmat @ rho_c0) 
        Z = top / bot 
        print(bot, top)
        z_norm = np.linalg.norm(Z) 
        if return_phase:
            z_phase = np.angle(Z) 
            return z_norm, z_phase
        else:
            return z_norm
    else:
        raise ValueError("Warning: The complex polarization cannot be calculated!")




def get_mu(h, nelec, beta, mu0=0):
    '''
    Optimize the mu value with respect to a given electron number.
    Args:
        h (2d array) : one-body Hamiltonian
        nelec (int) : target electron number 
        beta (float) : 1/temperature
    Kwargs:
        mu0 (float) : initial guess of chemical potential
    Returns:
        float, optimized mu.
    '''
    ew, _ = np.linalg.eigh(h)
    def fermi(mu):
        return 1./(1.+np.exp(beta*(ew-mu)))
    def func(mu):
        return (nelec - np.sum(fermi(mu)))**2
    mu = minimize(func, mu0, method="Powell").x[0]
    return mu