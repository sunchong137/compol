#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import ctypes
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring, direct_spin1
# from pyscf import lib 

libfci = direct_spin1.libfci

def contract_1e(f1e, fcivec, norb, nelec):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = fcivec.reshape(na,nb)
    t1 = numpy.zeros((norb,norb,na,nb), dtype=fcivec.dtype)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a,i,str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1[a,i,:,str1] += sign * ci0[:,str0]
    fcinew = numpy.dot(f1e.reshape(-1), t1.reshape(-1,na*nb))
    return fcinew.reshape(fcivec.shape)


def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    fcivec = numpy.asarray(fcivec, order='C')
    g2e_aa = ao2mo.restore(4, eri[0], norb)
    g2e_ab = ao2mo.restore(4, eri[1], norb)
    g2e_bb = ao2mo.restore(4, eri[2], norb)

    link_indexa, link_indexb = direct_spin1._unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert (fcivec.size == na*nb)
    ci1 = numpy.empty_like(fcivec)

    libfci.FCIcontract_uhf2e(g2e_aa.ctypes.data_as(ctypes.c_void_p),
                             g2e_ab.ctypes.data_as(ctypes.c_void_p),
                             g2e_bb.ctypes.data_as(ctypes.c_void_p),
                             fcivec.ctypes.data_as(ctypes.c_void_p),
                             ci1.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(norb),
                             ctypes.c_int(na), ctypes.c_int(nb),
                             ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                             link_indexa.ctypes.data_as(ctypes.c_void_p),
                             link_indexb.ctypes.data_as(ctypes.c_void_p))
    return ci1.view(direct_spin1.FCIvector)

def contract_2e_hubbard(u, fcivec, norb, nelec, opt=None):
    if isinstance(nelec, (int, numpy.number)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    u_aa, u_ab, u_bb = u

    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    fcivec = fcivec.reshape(na,nb)
    t1a = numpy.zeros((norb,na,nb))
    t1b = numpy.zeros((norb,na,nb))
    fcinew = numpy.zeros_like(fcivec)

    for addr, s in enumerate(strsa):
        for i in range(norb):
            if s & (1 << i):
                t1a[i,addr] += fcivec[addr]
    for addr, s in enumerate(strsb):
        for i in range(norb):
            if s & (1 << i):
                t1b[i,:,addr] += fcivec[:,addr]

    if u_aa != 0:
        # u * n_alpha^+ n_alpha
        for addr, s in enumerate(strsa):
            for i in range(norb):
                if s & (1 << i):
                    fcinew[addr] += t1a[i,addr] * u_aa
    if u_ab != 0:
        # u * n_alpha^+ n_beta
        for addr, s in enumerate(strsa):
            for i in range(norb):
                if s & (1 << i):
                    fcinew[addr] += t1b[i,addr] * u_ab
        # u * n_beta^+ n_alpha
        for addr, s in enumerate(strsb):
            for i in range(norb):
                if s & (1 << i):
                    fcinew[:,addr] += t1a[i,:,addr] * u_ab
    if u_bb != 0:
        # u * n_beta^+ n_beta
        for addr, s in enumerate(strsb):
            for i in range(norb):
                if s & (1 << i):
                    fcinew[:,addr] += t1b[i,:,addr] * u_bb
    return fcinew


def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    if not isinstance(nelec, (int, numpy.number)):
        nelec = sum(nelec)
    h1e_a, h1e_b = h1e
    h2e_aa = ao2mo.restore(1, eri[0], norb).copy()
    h2e_ab = ao2mo.restore(1, eri[1], norb).copy()
    h2e_bb = ao2mo.restore(1, eri[2], norb).copy()
    f1e_a = h1e_a - numpy.einsum('jiik->jk', h2e_aa) * .5
    f1e_b = h1e_b - numpy.einsum('jiik->jk', h2e_bb) * .5
    f1e_a *= 1./(nelec+1e-100)
    f1e_b *= 1./(nelec+1e-100)
    for k in range(norb):
        h2e_aa[:,:,k,k] += f1e_a
        h2e_aa[k,k,:,:] += f1e_a
        h2e_ab[:,:,k,k] += f1e_a
        h2e_ab[k,k,:,:] += f1e_b
        h2e_bb[:,:,k,k] += f1e_b
        h2e_bb[k,k,:,:] += f1e_b
    return (ao2mo.restore(4, h2e_aa, norb) * fac,
            ao2mo.restore(4, h2e_ab, norb) * fac,
            ao2mo.restore(4, h2e_bb, norb) * fac)


def make_hdiag(h1e, eri, norb, nelec, compress=False):
    neleca, nelecb = nelec 
    h1e_a = numpy.ascontiguousarray(h1e[0])
    h1e_b = numpy.ascontiguousarray(h1e[1])
    g2e_aa = ao2mo.restore(1, eri[0], norb)
    g2e_ab = ao2mo.restore(1, eri[1], norb)
    g2e_bb = ao2mo.restore(1, eri[2], norb)

    occslsta = occslstb = cistring.gen_occslst(range(norb), neleca)
    if neleca != nelecb:
        occslstb = cistring.gen_occslst(range(norb), nelecb)
    na = len(occslsta)
    nb = len(occslstb)

    hdiag = numpy.empty(na*nb)
    jdiag_aa = numpy.asarray(numpy.einsum('iijj->ij',g2e_aa), order='C')
    jdiag_ab = numpy.asarray(numpy.einsum('iijj->ij',g2e_ab), order='C')
    jdiag_bb = numpy.asarray(numpy.einsum('iijj->ij',g2e_bb), order='C')
    kdiag_aa = numpy.asarray(numpy.einsum('ijji->ij',g2e_aa), order='C')
    kdiag_bb = numpy.asarray(numpy.einsum('ijji->ij',g2e_bb), order='C')
    libfci.FCImake_hdiag_uhf(hdiag.ctypes.data_as(ctypes.c_void_p),
                             h1e_a.ctypes.data_as(ctypes.c_void_p),
                             h1e_b.ctypes.data_as(ctypes.c_void_p),
                             jdiag_aa.ctypes.data_as(ctypes.c_void_p),
                             jdiag_ab.ctypes.data_as(ctypes.c_void_p),
                             jdiag_bb.ctypes.data_as(ctypes.c_void_p),
                             kdiag_aa.ctypes.data_as(ctypes.c_void_p),
                             kdiag_bb.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(norb),
                             ctypes.c_int(na), ctypes.c_int(nb),
                             ctypes.c_int(neleca), ctypes.c_int(nelecb),
                             occslsta.ctypes.data_as(ctypes.c_void_p),
                             occslstb.ctypes.data_as(ctypes.c_void_p))
    return numpy.asarray(hdiag)


def kernel(h1e, eri, norb, nelec, ecore=0, target_e=None):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    if target_e is None:
        h2e = absorb_h1e(h1e, eri, norb, nelec, .5)
        ci0 = numpy.zeros((na, nb))
        ci0[0, 0] = 1

        def hop(c):
            hc = contract_2e(h2e, c, norb, nelec)
            return hc.reshape(-1)
        
        hdiag = make_hdiag(h1e, eri, norb, nelec)
        precond = lambda x, e, *args: x/(hdiag-e+1e-4)
        e, c = lib.davidson(hop, ci0.reshape(-1), precond)
    else:
        h1e = h1e - numpy.eye(norb)*target_e
        h2e = absorb_h1e(h1e, eri, norb, nelec, .5)
        ci0 = numpy.random.rand(na, nb) # random initial guess is better
        ci0 /= numpy.linalg.norm(ci0)

        def hop(c):
            hc = contract_2e(h2e, c, norb, nelec)
            hhc = contract_2e(h2e, hc, norb, nelec)
            return hhc.reshape(-1)   
        hdiag = make_hdiag(h1e, eri, norb, nelec) ** 2
        precond = lambda x, e, *args: x/(hdiag-e+1e-4)
        e, c = lib.davidson(hop, ci0.reshape(-1), precond)  

    return e+ecore, c


# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, opt=None):
    link_index = cistring.gen_linkstr_index(range(norb), nelec//2)
    na = cistring.num_strings(norb, nelec//2)
    fcivec = fcivec.reshape(na,na)
    rdm1 = numpy.zeros((norb,norb))
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in link_index[str0]:
            rdm1[a,i] += sign * numpy.dot(fcivec[str1],fcivec[str0])
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in link_index[str0]:
            rdm1[a,i] += sign * numpy.dot(fcivec[:,str1],fcivec[:,str0])
    return rdm1

# dm_pq,rs = <|p^+ q r^+ s|>
def make_rdm12(fcivec, norb, nelec, opt=None):
    link_index = cistring.gen_linkstr_index(range(norb), nelec//2)
    na = cistring.num_strings(norb, nelec//2)
    fcivec = fcivec.reshape(na,na)

    rdm1 = numpy.zeros((norb,norb))
    rdm2 = numpy.zeros((norb,norb,norb,norb))
    for str0, tab in enumerate(link_index):
        t1 = numpy.zeros((na,norb,norb))
        for a, i, str1, sign in link_index[str0]:
            t1[:,i,a] += sign * fcivec[str1,:]

        for k, tab in enumerate(link_index):
            for a, i, str1, sign in tab:
                t1[k,i,a] += sign * fcivec[str0,str1]

        rdm1 += numpy.einsum('m,mij->ij', fcivec[str0], t1)
        # i^+ j|0> => <0|j^+ i, so swap i and j
        rdm2 += numpy.einsum('mij,mkl->jikl', t1, t1)
    return reorder_rdm(rdm1, rdm2)


def reorder_rdm(rdm1, rdm2, inplace=True):
    '''reorder from rdm2(pq,rs) = <E^p_q E^r_s> to rdm2(pq,rs) = <e^{pr}_{qs}>.
    Although the "reoredered rdm2" is still in Mulliken order (rdm2[e1,e1,e2,e2]),
    it is the true 2e DM (dotting it with int2e gives the energy of 2e parts)
    '''
    nmo = rdm1.shape[0]
    if inplace:
        rdm2 = rdm2.reshape(nmo,nmo,nmo,nmo)
    else:
        rdm2 = rdm2.copy().reshape(nmo,nmo,nmo,nmo)
    for k in range(nmo):
        rdm2[:,k,k,:] -= rdm1
    return rdm1, rdm2


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]
    mol.basis = 'sto-3g'
    mol.build()

    m = scf.RHF(mol)
    m.kernel()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron - 2
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.kernel(m._eri, m.mo_coeff, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)

    e1 = kernel(h1e, eri, norb, nelec)
    print(e1, e1 - -7.9766331504361414)
