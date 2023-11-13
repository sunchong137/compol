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
Standard and disordered Hubbard model.
'''
import numpy as np
import logging
from pyscf import gto, scf, ao2mo, fci
from compol import helpers

def hubham_1d(nsite, U, pbc=True, noisy=False, max_w=1.0, spin=0):
    '''
    Return 1D Hubbard model Hamiltonians (h1e and h2e).
    Unit: hopping amplitude t.
    Args:
        nsite : int, number of sites.
        U: double, Hubbard U/t value.
        pbc: boolean, if True, the system obeys periodic boundary condition.
    Returns:
        2D array: h1e
        4D array: h2e
    '''
    h1e = np.zeros((nsite, nsite))
    h2e = np.zeros((nsite,)*4)

    for i in range(nsite-1):
        h1e[i, (i+1)] = -1.
        h1e[(i+1), i] = -1.
        h2e[i,i,i,i] = U
    h2e[-1,-1,-1,-1] = U

    if pbc:
        # assert nsite%4 == 2, "PBC requires nsite = 4n+2!"
        h1e[0, -1] = -1.
        h1e[-1, 0] = -1.

    if noisy:
        if spin == 0:
            noise = (np.random.rand(nsite)*2-1) * max_w
            noise = np.diag(noise)
            h1e += noise 
            return h1e, h2e
        elif spin == 1:
            noise1 = (np.random.rand(nsite)*2-1) * max_w
            noise2 = (np.random.rand(nsite)*2-1) * max_w
            return [h1e+np.diag(noise1), h1e+np.diag(noise2)], h2e
        else:
            raise ValueError("spin has to be 0 or 1.")
    else:
        return h1e, h2e


def hamhub_2d(nx, ny, U, pbc=True):
    if nx == 1:
        return hubham_1d(ny, U, pbc)
    if ny == 1:
        return hubham_1d(nx, U, pbc)

    nsite = nx * ny
    h1e = np.zeros((nsite, nsite))
    for i in range(ny-1):
        for j in range(nx-1):
            idx = i*nx + j 
            idx_r = i*nx + j + 1
            idx_d = (i+1)*nx + j
            h1e[idx, idx_r] = h1e[idx_r, idx] = -1
            h1e[idx, idx_d] = h1e[idx_d, idx] = -1

        h1e[(i+1)*nx-1, (i+2)*nx-1] = h1e[(i+2)*nx-1, (i+1)*nx-1] = -1

    dn = (ny-1) * nx
    for j in range(nx-1):
        h1e[dn+j, dn+j+1] = h1e[dn+j+1, dn+j] = -1


    eri = np.zeros((nsite, )*4)
    for i in range(nsite):
        eri[i,i,i,i] = U
    if pbc:
        # down-up
        for i in range(nx):
            h1e[i, dn+i] = h1e[dn+i, i] = -1
            
        # right-left
        for j in range(ny):
            h1e[nx*j, nx*(j+1)-1] = h1e[nx*(j+1)-1, nx*j] = -1

    return h1e, eri


def hubham_noisy_1d(nsite, U, pbc=True, max_w=1.0):
    '''
    Add diagonal noise to the original Hubbard model.
    The noise is a uniform distribution in [-max_w, max_w]
    '''
    h1e, h2e = hubham_1d(nsite, U, pbc)
    # generate uniform distribution
    noise = (np.random.rand(nsite)*2-1) * max_w
    noise = np.diag(noise)
    return h1e+noise, h2e


def hubham_spinless_1d(nsite, V, pbc=True):
    '''
    Spinless Hubbard model with nearest neighbor e-e interaction.
    
        :math: H = t \sum_i c^\dag_i c_{i+1} + V \sum_i n_i n_i+1
    '''
    h1e = np.zeros((nsite, nsite))
    h2e = np.zeros((nsite,)*4)

    for i in range(nsite-1):
        h1e[i, i+1] = -1.
        h1e[i+1, i] = -1.
        h2e[i, i, i+1, i+1] = V
        h2e[i+1, i+1, i, i] = V
    if pbc:
        h1e[0, -1] = h1e[-1, 0] = -1.
        h2e[0, 0, -1, -1] = h2e[-1, -1, 0, 0] = V 
    return h1e, h2e


def hubham_spinless_noisy_1d(nsite, V, pbc=True, max_w=1.0):
    '''
    Add noise to the spinless Hubbard model.
    '''
    h1e, h2e = hubham_spinless_1d(nsite, V, pbc)
    # generate uniform distribution
    noise = (np.random.rand(nsite)*2-1) * max_w
    noise = np.diag(noise)
    return h1e+noise, h2e

def hubbard_mf(nsite, U, spin=0, nelec=None, pbc=True, filling=1.0):
    '''
    Mean-field of Hubbard model.
    Args:
        nsite: int, number of sites
        U: positive float, Hubbard U/t
    Kwargs:
        spin: 0 (RHF) or 1 (UHF)
        nelec: int, number of electrons.
        pbc: boolean, periodic boundary condition
        filling: float, nelec/nsite. value 1 corresponds to half-filling.
    Returns:
        float: HF energy 
        numpy array of size (nspin, nsite, nsite), the MO coefficients.
    '''
    # nsite need to be 4n + 2, otherwise there is degeneracy between HOMO and LUMO
    mol = gto.M()

    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
    if abs(nelec - nsite * filling) > 1e-2:
        logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer number of electrons!".format(filling, nelec/nsite))
    
    mol.nelectron = nelec
    mol.nao = nsite
    h1e, eri = hubham_1d(nsite, U, pbc=pbc)
    
    if spin == 0:
        mf = scf.RHF(mol)
    elif spin == 1:
        mf = scf.UHF(mol)
    else:
        raise ValueError("Spin has to be 0 or 1!")

    mf.get_hcore = lambda *args: h1e 
    mf.get_ovlp = lambda *args: np.eye(nsite)
    mf._eri = ao2mo.restore(8, eri, nsite)
    mol.incore_anyway = True

    if spin == 1:
        helpers.run_stab_mf(mf)
    else:
        mf.kernel()
    return mf

def hubbard_spinless_mf(nsite, V, nelec=None, pbc=True, filling=1.0):
    '''
    Mean-field for spinless Hubbard model.
    '''
    mol = gto.M()
    filling /= 2 # spinless 
    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
    if abs(nelec - nsite * filling) > 1e-2:
        logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer number of electrons!".format(filling, nelec/nsite))
    

    mol.nelectron = nelec
    mol.nao = nsite
    mol.spin = nelec
    h1e, eri = hubham_spinless_1d(nsite, V, pbc=pbc)
    
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h1e 
    mf.get_ovlp = lambda *args: np.eye(nsite)
    mf._eri = ao2mo.restore(8, eri, nsite)
    mol.incore_anyway = True
    mf.kernel()

    return mf


def hubbard_fci(nsite, U, nelec=None, pbc=True, filling=1.0):
    
    h1e, eri = hubham_1d(nsite, U, pbc)

    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
    if abs(nelec - nsite * filling) > 1e-2:
        logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer number of electrons!".format(filling, nelec/nsite))
    
    # initial guess
    try:
        na, nb = nelec 
    except:
        na = nelec//2
        nb = nelec - na

    if na == nb:
        cisolver = fci.direct_spin0
    else:
        cisolver = fci.direct_spin1
        nelec = (na, nb)
    # len_a = int(special.comb(nsite, na) + 1e-10)
    # len_b = int(special.comb(nsite, nb) + 1e-10)
    # ci0 = np.random.rand(len_a, len_b)
    # ci0 = np.ones((len_a, len_b)) / np.sqrt(len_a*len_b)

    e, c = cisolver.kernel(h1e, eri, nsite, nelec)
    return e, c


def hubbard_spinless_fci(nsite, V, nelec=None, pbc=True, filling=1.0):

    h1e, eri = hubham_spinless_1d(nsite, V, pbc)
    h1_0 = np.zeros_like(h1e)
    h2_0 = np.zeros_like(eri)

    filling /= 2
    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
    if abs(nelec - nsite * filling) > 1e-2:
        logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer number of electrons!".format(filling, nelec/nsite))

    cisolver = fci.direct_uhf
    h1_uhf = (h1e, h1_0) # same as h1_uhf = (h1e, h1e)
    h2_uhf = (eri, h2_0, h2_0)  # same as h2_uhf = (eri, eri, eri)
    e, c = cisolver.kernel(h1_uhf, h2_uhf, nsite, (nelec, 0)) 
    # c is of shape (nstr, 1)
    return e, c.ravel()


def hubbard_fci_from_mf(mf):
    '''
    This one is faster because the initial guess is better.
    '''
    cisolver = fci.FCI(mf)
    ci_energy, ci = cisolver.kernel()
    
    return ci_energy, ci
