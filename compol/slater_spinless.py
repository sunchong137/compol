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
Evaluate complex polarization based on Slater determinants.
1D model system with site basis only, 
'''

import numpy as np 
from compol import slater

Pi = np.pi

def gen_zmat_site(L, x0, T=0, Tmin=1e-2):
    '''
    Generate the Z operator in the site basis for MO coeffs.
    '''
    return slater.gen_zmat_site(L, x0, T=T, Tmin=Tmin)

def gen_det(mo_coeff, occ):
    '''
    Given the occupations (occ), generate the Slater determinant from the MO coefficients.
    Args:
        mo_coeff: 2D array of size (nsite, nsite)
    '''

    idx = np.nonzero(occ)[0]
    sdets = np.copy(mo_coeff[:, idx])
    return sdets

def z_sdet(L, det, x0=0.0):
    '''
    Applies complex polarzation Z onto a Slater determinant represented
    by det. 
    Even though Z is an L-body operator, it operating on a Slater determinant
    gives another Slater determinant with the AO basis rotated.
    Args:
        L: integer, number of sites.
        det: numpy array of size (L, Nocc)
    kwargs:
        x0: origin.
    returns:
        numpy array of size (L, Nocc), the new Slater determinant.
    '''

    Z = gen_zmat_site(L, x0)
    mo_new = np.dot(Z, det)
    return mo_new

def ovlp_det(sdet1, sdet2, ao_ovlp=None):
    '''
    Evaluate the overlap between two determinants.
    Args:
        det1: numpy array of size (spin, L, Nocc)
        det2: numpy array of size (spin, L, Nocc)
    Kwargs:
        ao_ovlp: overlap matrix of AO orbitals, always identity for site basis.
    Returns:
        a complex number: overlap between two determinants.
    '''
        
    if ao_ovlp is None:
        ovlp = np.linalg.det(np.dot(sdet1.T.conj(), sdet2))
    else:
        ovlp = np.linalg.det(sdet1.T.conj() @ ao_ovlp @ sdet2)
    return ovlp

def det_z_det(L, sdet, x0=0.0, normalize=False):
    '''
    Evaluate <det | Z | det> / <det | det>
    Returns:
        A complex number
    '''
    sdet2 = z_sdet(L, sdet, x0=x0)
    Z = ovlp_det(sdet, sdet2) 
    if normalize:
        Z /= ovlp_det(sdet, sdet)
    return np.linalg.norm(Z)