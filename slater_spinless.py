'''
Evaluate complex polarization based on Slater determinants.
1D model system with site basis only, 
'''

import numpy as np 
import slater_uhf

Pi = np.pi

def gen_zmat_site(L, x0):
    '''
    Generate the Z operator in the site basis for MO coeffs.
    '''
    return slater_uhf.gen_zmat_site(L, x0)

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

def det_z_det(L, sdet, x0=0.0):
    '''
    Evaluate <det | Z | det> / <det | det>
    Returns:
        A complex number
    '''
    return slater_uhf.det_z_det(L, sdet, x0)
