'''
Evaluate complex polarization based on Slater determinants.
1D model system with site basis only, 
'''

import numpy as np 

Pi = np.pi

def z_sdet(L, mo_coeff, x0=0.0):
    '''
    Applies complex polarzation Z onto a Slater determinant represented
    by mo_coeff. 
    Even though Z is an L-body operator, it operating on a Slater determinant
    gives another Slater determinant with the AO basis rotated.
    Args:
        L: integer, number of sites.
        mo_coeff: numpy array of size (spin, L, Nocc)
    kwargs:
        x0: origin.
    returns:
        numpy array of size (spin, L, Nocc), the new Slater determinant.
    '''
    ndim = mo_coeff.ndim
    pos = np.arange(L) + x0 
    Z = np.exp(2.j * Pi * pos / L)
    Z = np.diag(Z)
    if ndim == 2: # RHF
        mo_new = np.dot(Z, mo_coeff)
    elif ndim == 3:
        mo_new = np.array([np.dot(Z, mo_coeff[0]), np.dot(Z, mo_coeff[1])])
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
    ndim = sdet1.ndim
    if ao_ovlp is None:
        if ndim == 2:
            ovlp = np.linalg.det(np.dot(sdet1.T.conj(), sdet2))
        elif ndim == 3:
            ovlp1 = np.linalg.det(np.dot(sdet1[0].T.conj(), sdet2[0]))
            ovlp2 = np.linalg.det(np.dot(sdet1[1].T.conj(), sdet2[1]))
            ovlp = ovlp1 * ovlp2 
    else:
        if ndim == 2:
            ovlp = np.linalg.det(np.dot(np.dot(sdet1.T.conj(), ao_ovlp), sdet2))
        elif ndim == 3:
            ovlp1 = np.linalg.det(np.dot(np.dot(sdet1[0].T.conj(), ao_ovlp), sdet2[0]))
            ovlp2 = np.linalg.det(np.dot(np.dot(sdet1[1].T.conj(), ao_ovlp), sdet2[1]))
            ovlp = ovlp1 * ovlp2 

    return ovlp

def det_z_det(L, sdet, x0=0.0):
    '''
    Evaluate <det | Z | det> / <det | det>
    '''
    sdet2 = z_sdet(L, sdet, x0=x0)
    Z = ovlp_det(sdet, sdet2) / ovlp_det(sdet, sdet)
    return Z