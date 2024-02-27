import random
import string
import numpy as np
import scipy



def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def break_kmf_sym(dm0, shift=None):
    #TODO fix the negative output
    nspin = dm0[0]
    ncell = dm0.shape[1]
    if shift is None:
        shift = dm0[0,0,0,0] - 1e-2
    if shift > dm0[0,0,0,0]:
        shift = dm0[0,0,0,0] - 1e-4
    
    for i in range(ncell):
        dm0[0,i,0,0] += shift
        dm0[1,i,0,0] -= shift
    return dm0 

def break_kmf_sym_2x2(dm0, shift=None):
    #TODO fix the negative output
    nspin = dm0.shape[0]
    ncell = dm0.shape[1]
    norb = dm0[0,0].shape[-1]
    nbasis = int(norb//4)
    if shift is None:
        shift = dm0[0,0,0,0] - 1e-2
    if shift > dm0[0,0,0,0]:
        shift = dm0[0,0,0,0] - 1e-4
    
    for k in range(ncell):
        for i in range(2):
            for j in range(2):
                idx = (i*2+j)*nbasis
                if nbasis > 1:
                    dm0[0,k,idx,idx] += dm0[0,k,idx+1,idx+1]
                    dm0[1,k,idx,idx] += dm0[1,k,idx+1,idx+1]
                    dm0[0,k,idx+1,idx+1] -= dm0[0,k,idx+1,idx+1]
                    dm0[1,k,idx+1,idx+1] -= dm0[1,k,idx+1,idx+1]
                dm0[0,k,idx,idx] += shift*(-1)**(i+j)
                dm0[1,k,idx,idx] -= shift*(-1)**(i+j)
    return dm0 



def slice_to_full(m):
    '''
    turn sliced matrix back to the full matrix
    '''
    data_type = m.dtype.name
    nscsite = m.shape[-1]
    ncell = m.shape[-3]
    norb = nscsite *  ncell
    mat = np.copy(m)
    if mat.ndim == 3:
        mat[0] = mat[0] / 2.
        mat_n = np.zeros((norb, norb), dtype = data_type)
        for n in range(ncell):
            mat_n[nscsite*n : nscsite*(n+1), nscsite*n:] = np.hstack(mat[:(ncell-n)])
        mat_n += mat_n.conj().T

    elif mat.ndim == 4:
        spin = mat.shape[0]
        mat[:,0,...] = mat[:,0,...] / 2.
        mat_n = np.zeros((spin, norb, norb), dtype = data_type)
        for s in range(spin):
            for n in range(ncell):
                mat_n[s, nscsite*n:nscsite*(n+1), nscsite*n:] = np.hstack(mat[s, :(ncell-n)])
        mat_n = mat_n + mat_n.transpose(0,2,1).conj()

    return mat_n

def exp_mat(m):
    if m.ndim == 2:
        return scipy.linalg.expm(m)
    else:
        return np.array([scipy.linalg.expm(m[0]), scipy.linalg.expm(m[1])])
        
def vcor_2d_minao(afm, nsite, nx, ny):
    zmat = np.zeros((2, nsite, nsite))
    nbase = int(nsite//(nx*ny))
    for s in range(2):
        for i in range(nx):
            for j in range(ny):
                idx = (i*nx+j)# *nbase
                zmat[s, idx, idx] = (-1)**(i+j+s) * afm
                
    return zmat


if __name__ == "__main__":

    # test id_generator
    print(id_generator(4))
    print(id_generator(8))

    # test slice to full
    #m3 = np.random.rand(3,2,2)
    #m3 = m3 + m3.transpose(0,2,1)
    #print(m3)
    #print(slice_to_full(m3))
    m = np.random.rand(2,3,2,2)
    m = m + m.transpose(0,1,3,2)
    print(m)
    m_n = slice_to_full(m)
    print(m_n)
    
