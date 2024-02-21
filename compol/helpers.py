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

import numpy as np
from pyscf import ao2mo
from pyscf.fci import cistring

def run_stab_mf(mf):
    '''
    Run mf with stabalizer.
    '''
    mf.kernel()
    mo1 = mf.stability()[0]
    init = mf.make_rdm1(mo1, mf.mo_occ)                                                 
    mf.kernel(init)

def rotate_ham(mf):
    '''
    Rotate the Hamiltonian from atomic orbitals to molecular orbitals.
    '''
    h1e = mf.get_hcore()
    norb = mf.mol.nao
    eri = mf._eri
    mo_coeff = mf.mo_coeff
    # aaaa, aabb, bbbb
    Ca, Cb = mo_coeff[0], mo_coeff[1]
    aaaa = (Ca,)*4
    bbbb = (Cb,)*4
    aabb = (Ca, Ca, Cb, Cb)

    h1_mo = np.array([Ca.T@h1e@Ca, Cb.T@h1e@Cb])
    h2e_aaaa = ao2mo.incore.general(eri, aaaa, compact=False).reshape(norb, norb, norb, norb)
    h2e_bbbb = ao2mo.incore.general(eri, bbbb, compact=False).reshape(norb, norb, norb, norb)
    h2e_aabb = ao2mo.incore.general(eri, aabb, compact=False).reshape(norb, norb, norb, norb)
    h2_mo = np.array([h2e_aaaa, h2e_aabb, h2e_bbbb])

    return h1_mo, h2_mo

def rotate_ham_spinless(mf):
    '''
    Rotate the Hamiltonian from atomic orbitals to molecular orbitals.
    '''
    h1e = mf.get_hcore()
    norb = mf.mol.nao
    eri = mf._eri
    mo_coeff = mf.mo_coeff
    # aaaa, aabb, bbbb
    C = mo_coeff[0]
    aaaa = (C,)*4

    h1_mo = np.array([C.T@h1e@C, h1e*0])
    h2e_aaaa = ao2mo.incore.general(eri, aaaa, compact=False).reshape(norb, norb, norb, norb)
    h2_mo = np.array([h2e_aaaa, h2e_aaaa*0, h2e_aaaa*0])
    return h1_mo, h2_mo


def ao2mo_spinless(h1e, g2e, mo_coeff):
    '''
    Transform the Hamiltonians in the site basis to the MO basis.
    Args:
        h1e: 2D array
        g2e: 4D array
        mo_coeff: 3D array
    Returns:
        h1e_mo
        g2e_mo
    '''
    moc = mo_coeff[0] # only alpha spin
    nao = moc.shape[-1]
    h1e_a = moc.conj().T @ h1e @ moc 
    g2e_a = ao2mo.incore.general(g2e, (moc,)*4, compact=False).reshape(nao, nao, nao, nao)
    # return h1e_a, g2e_a
    return np.array([h1e_a, h1e_a]), np.array([g2e_a, g2e_a, g2e_a]) 

def ao2mo_uhf(h1e, g2e, mo_coeff):
    h1e_mo = np.array(
        [mo_coeff[0].T @ h1e[0] @ mo_coeff[0], mo_coeff[1].T @ h1e[0] @ mo_coeff[1]]
    )
    h2aa = ao2mo.kernel(g2e[0], (mo_coeff[0], mo_coeff[0], mo_coeff[0], mo_coeff[0]))
    h2ab = ao2mo.kernel(g2e[1], (mo_coeff[0], mo_coeff[0], mo_coeff[1], mo_coeff[1]))
    h2bb = ao2mo.kernel(g2e[2], (mo_coeff[1], mo_coeff[1], mo_coeff[1], mo_coeff[1]))
    h2e_mo = np.array([h2aa, h2ab, h2bb])
    return h1e_mo, h2e_mo 

def contract_1e_uhf(f1e, fcivec, norb, nelec):
    '''
    Evaluate h|c>, where h is a one-body operator and |c> is a vector.
    Adapted from PySCF.
    Args: 
        f1e: list of 2D arrays, one-body Hamiltonian of spin-up and spin-down separately.
        fcivec: 1D or 2D array, FCI vector.
        norb: int, number of orbitals.
        nelec: int or tuple, number of electrons.
    Returns:
        1D or 2D array, a new FCI vector.
    '''
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = fcivec.reshape(na,nb)
    t1a = np.zeros((norb,norb,na,nb), dtype=fcivec.dtype)
    t1b = np.zeros((norb,norb,na,nb), dtype=fcivec.dtype)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1a[a,i,str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1b[a,i,:,str1] += sign * ci0[:,str0]
    fcinew = f1e[0].reshape(-1) @ t1a.reshape(-1,na*nb) + f1e[1].reshape(-1) @ t1b.reshape(-1,na*nb)
    return fcinew.reshape(fcivec.shape)


def contract_1e_onespin(f1e, fcivec, norb, nelec, spin="a"):
    '''
    Only apply the one-body operator onto one spin.
    Args: 
        f1e: 2D array, one-body Hamiltonian of spin-up OR spin-down.
        fcivec: 1D or 2D array, FCI vector.
        norb: int, number of orbitals.
        nelec: int or tuple, number of electrons.
    Kwargs:
        spin: char, "a" - spin-up, "b" - spin down.
    Returns:
        1D or 2D array, a new FCI vector.
    '''
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = fcivec.reshape(na,nb)
    if spin == "a":
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        t1a = np.zeros((norb,norb,na,nb), dtype=fcivec.dtype)
        for str0, tab in enumerate(link_indexa):
            for a, i, str1, sign in tab:
                t1a[a,i,str1] += sign * ci0[str0]
        fcinew = f1e.reshape(-1) @ t1a.reshape(-1,na*nb)
    elif spin == "b":
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        t1b = np.zeros((norb,norb,na,nb), dtype=fcivec.dtype)
        for str0, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                t1b[a,i,:,str1] += sign * ci0[:,str0]
        fcinew = f1e.reshape(-1) @ t1b.reshape(-1,na*nb)
    else:
        raise ValueError("spin can only be 'a' or 'b'!")
    return fcinew.reshape(fcivec.shape)

def contract_1e_spinless():
    # TODO
    pass