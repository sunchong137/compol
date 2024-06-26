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

Pi = np.pi

def gen_zmat_site(L, x0):
    '''
    Generate the matrix for Z operator in the site basis for MO coeffs.
    '''
    pos = np.arange(L) + x0 
    Z = np.exp(2.j * Pi * pos / L)
    return np.diag(Z)


def gen_det(mo_coeff, occ):
    '''
    Given the occupations (occ), generate the Slater determinant from the MO coefficients.
    Args:
        mo_coeff: 2D or 3D array of the MO coefficients.
        occ: 1D or 2D array of elements 0 or 1, the occupation of each MO.
    '''
    occ = np.asarray(occ)
    try:
        ndim = mo_coeff.ndim
    except:
        ndim = 3
    if ndim > 2: # UHF
        idx_up = np.nonzero(occ[0])[0]
        idx_dn = np.nonzero(occ[1])[0]
        det_up = mo_coeff[0][:, idx_up]
        det_dn = mo_coeff[1][:, idx_dn]
        dets = [det_up, det_dn] # not array because shapes of up and dn can be diff
    else: # RHF
        idx = np.nonzero(occ)[0]
        dets = np.copy(mo_coeff[:, idx])#.reshape(norb, nocc)
    return dets


def z_sdet(L, sdet, x0=0.0):
    '''
    Applies complex polarzation Z onto a Slater determinant represented
    by sdet. 
    Even though Z is an L-body operator, it operating on a Slater determinant
    gives another Slater determinant with the AO basis rotated.
    Args:
        L: integer, number of sites.
        sdet: numpy array of size (spin, L, Nocc)
    kwargs:
        x0: origin.
    returns:
        numpy array of size (spin, L, Nocc), the new Slater determinant.
    '''
    try:
        ndim = sdet.ndim
    except: # list or tuple
        ndim = 3

    Z = gen_zmat_site(L, x0)

    if ndim == 2: # RHF
        mo_new = Z @ sdet
    elif ndim == 3:
        mo_new = np.array([Z @ sdet[0], Z @ sdet[1]])
    else:
        raise ValueError("The MO coefficient matrix is an array.")
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
    try:
        ndim = sdet1.ndim
    except: # list or tuple
        ndim = 3
        
    if ao_ovlp is None:
        if ndim == 2:
            ovlp = np.linalg.det(sdet1.T.conj() @ sdet2)
            ovlp = ovlp ** 2
        elif ndim == 3:
            ovlp1 = np.linalg.det(sdet1[0].T.conj() @ sdet2[0])
            ovlp2 = np.linalg.det(sdet1[1].T.conj() @ sdet2[1])
            ovlp = ovlp1 * ovlp2 
    else:
        if ndim == 2:
            ovlp = np.linalg.det(sdet1.T.conj() @ ao_ovlp @ sdet2)
            ovlp = ovlp * ovlp
        elif ndim == 3:
            ovlp1 = np.linalg.det(sdet1[0].T.conj() @ ao_ovlp @ sdet2[0])
            ovlp2 = np.linalg.det(sdet1[1].T.conj() @ ao_ovlp @ sdet2[1])
            ovlp = ovlp1 * ovlp2 
    return ovlp


def det_z_det(L, sdet, x0=0.0, normalize=False, return_phase=False):
    '''
    Evaluate <det | Z | det> 
    Args:
        L: int, number of sites
        sdet: 2D or 3D array, MO coefficients of the Slater determinant.
    Kwargs:
        x0: float, origin
    Returns:
        A complex number
    '''
    sdet2 = z_sdet(L, sdet, x0=x0)
    Z = ovlp_det(sdet, sdet2) 
    if normalize:
        Z /= ovlp_det(sdet, sdet)
        
    z_norm = np.linalg.norm(Z) 
    if return_phase:
        z_phase = np.angle(Z) 
        return z_norm, z_phase
    else:
        return z_norm