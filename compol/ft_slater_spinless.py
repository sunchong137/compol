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

def det_z_det(L, mf, T, x0=0, Tmin=1e-2, mu=None, return_phase=False):
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
    fock = mf.get_fock()[0]
    if mu is None:
        nelec = np.sum(mf.nelec) 
        mu = get_mu(fock, nelec, beta, mu0=0)
    mu_mat = np.eye(L) * mu
    zmat = gen_zmat_site(L, x0) 
    rho = np.eye(L*2)
    # rho[:L, :L] = rdm1_ft(mf) 
    rho[:L, :L] = sla.expm(-beta * (fock-mu_mat))
    C0 = np.zeros((2*L, L))
    C0[:L] = np.eye(L)
    C0[L:] = np.eye(L)

    # rescale C0 for stability 
    rho_c0 = rho @ C0 
    top = np.linalg.det(C0.T @ zmat @ rho_c0) 
    bot = np.linalg.det(C0.T @ rho_c0)
    print(bot, top)
    Z = top / bot 
    z_norm = np.linalg.norm(Z) 
    if return_phase:
        z_phase = np.angle(Z) 
        return z_norm, z_phase
    else:
        return z_norm


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