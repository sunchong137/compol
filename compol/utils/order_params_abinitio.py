import numpy as np
import scipy

def complex_polarization_gs(zparam, mo_coeff=None, rho=None, imag_tol=1e-10):
    '''
    args:
        zparam : (spin, ncell, nao) complex matrix
            parameters of the z-matrix
        mo_coeff: (spin, nmo, norb) matrix
            occupied mo coefficients in real space
        rho: (spin, norb, norb) matrix
            1-particle density matrix in real space
    '''

    spin = zparam.shape[0] # only for UHF

    if (mo_coeff is None) and (rho is None):
        raise ValueError("Please provide at least one value of mo_coeff and rho!")

    if mo_coeff is None:
        # diagonalize rho to get mo_coeff
        mo_occ = []
        norb = rho.shape[-1]
        for s in range(spin):
            ew, ev = np.linalg.eigh(rho[s])
            occ = np.where(ew > 0.4)[0] # use 0.9 instead of 0 for low T case
            mo_occ.append(np.copy(ev[:,occ]))
    else:
        mo_occ = mo_coeff
        norb = mo_occ.shape[-2]

    zmat = expand_zmat(zparam, norb)
    exp_zmat = exp_mat(zmat) 

    n_occ_a = mo_occ[0].shape[-1]
    n_occ_b = mo_occ[1].shape[-1]
    if (n_occ_a == n_occ_b):
        mo_occ = np.asarray(mo_occ)
        cp = ovlp_det(mo_occ, exp_zmat)
    else:
        cpa = ovlp_det_rhf(mo_occ[0], exp_zmat[0])
        cpb = ovlp_det_rhf(mo_occ[1], exp_zmat[1])
        cp = cpa*cpb

    if (cp.imag < imag_tol):
        cp = cp.real
    else:
        print("complex polarization has non-zero imaginary part = %0.6f!"%cp.imag)
    return cp


def complex_polarization_ft(zparam, h, beta, mu=0., imag_tol=1e-8, exp_tol=10):
    '''
        Calculate the finite temperature complex polarization using ancillas.
    '''
    hcore = np.copy(h)
    norb = hcore.shape[-1]
    spin = hcore.shape[0]

    zmat = expand_zmat(zparam, norb)
    # add mu to hcore
    for s in range(spin):
        hcore[s] = hcore[s] - np.eye(norb) * mu

    ew, _ = np.linalg.eigh(hcore)
    e0 = min(ew[0,0], ew[1,0])
    kmat = -1.* beta * hcore

    # construct thermal determinant (physical site + ancilla)
    # norb particles, 2*norb orbitals
    # first norb orbitals: physical sites, last norb orbitals: ancillas
    D = np.zeros((spin, 2*norb, norb))
    for s in range(spin):
        for i in range(norb):
            D[s, i, i] = 1.
            D[s, norb+i, i] = 1.

    # scan the factor to avoid overflow/underflow
    fac = 0.1
    step = 0.9
    max_iter = 100
    conv = False
    for i in range(max_iter):
        D *= fac
        # calculate partition function
        thermal_h = make_thermal_op(kmat)
        exp_h = exp_mat(thermal_h)
        part_func = ovlp_det(D, exp_h)
        if part_func > 1e200:
            fac *= step
            continue
        elif part_func < 1e-200:
            fac /= step
            continue
        else:
            # calculate position value
            thermal_z = make_thermal_op(zmat)
            exp_z = exp_mat(thermal_z)
            pos = ovlp_det(D, exp_z @ exp_h)
            cp = pos/part_func
            conv = True
            break
    if not conv:
        print("The complex polarization cannot be calculated!")
        cp = -1.

    if cp.imag < imag_tol:
        cp = cp.real
        
    return cp
    

def ovlp_det(D, mat):
    D1 = mat @ D
    ovlp = np.linalg.det(np.transpose(D.conj(), (0,2,1)) @ D1)
    return np.prod(ovlp) 

def ovlp_det_rhf(D, mat):
    D1 = np.dot(mat, D)
    ovlp = np.linalg.det(np.dot(D.conj().T, D1))
    return ovlp
   
def make_thermal_op(op):
    '''
    expand the 1-body operator in physical space to the physical+ancilla space.
    '''
    norb = op.shape[-1]
    spin = op.shape[0]
    thermal_op = np.zeros((spin, 2*norb, 2*norb), dtype=op.dtype.name)
    thermal_op[:, :norb, :norb] = op
    return thermal_op
    
    
def expand_zmat(z, norb):
    # get full z matrix
    spin = z.shape[0]
    zmat = np.zeros((spin, norb, norb), dtype=np.complex128)
    if (z.ndim == 3):
        for s in range(spin):
            zmat[s] = np.diag(z[s].ravel())
    else:
        raise Exception("Non-diagonal z-matrix is not implemented!")
    return zmat
   

def exp_mat(m):
    if m.ndim == 2:
        return scipy.linalg.expm(m)
    else:
        return np.array([scipy.linalg.expm(m[0]), scipy.linalg.expm(m[1])])