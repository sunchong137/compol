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

def det_z_det(L, fock, T, x0=0, Tmin=1e-2):
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
        raise ValueError("Temperature value is too small!")
        # TODO return ground state value
    Z = gen_zmat_site(L, x0) 
    rho = np.zeros((L*2, L*2)) 
    rho[:L, :L] = sla.expm(-1*fock/T)
    rho[L:, L:] = np.eye(L)
    C0 = np.zeros((2*L, L))
    C0[:L] = np.eye(L) 
    C0[L:] = np.eye(L) 
    
    rho_c0 = rho @ C0 
    top = C0.T @ Z @ rho_c0 
    bot = C0.T @ rho_c0  

    z_val = top / bot 
    return np.linalg.norm(z_val)