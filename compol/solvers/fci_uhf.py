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
Adapted from PySCF's fci_slow.py
Added features:
- UHF
- Target the middle of the energy spectrum.
'''

import numpy
import ctypes
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring, direct_spin1, rdm
# from pyscf import lib 

libfci = direct_spin1.libfci

def contract_1e(f1e, fcivec, norb, nelec):
    neleca, nelecb = _unpack_nelec(nelec)
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
    neleca, nelecb = _unpack_nelec(nelec)
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
    neleca, nelecb = _unpack_nelec(nelec)
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


def make_rdm1s(fcivec, norb, nelec, link_index=None):
    r'''Spin separated 1-particle density matrices.
    The return values include two density matrices: (alpha,alpha), (beta,beta)

    dm1[p,q] = <q^\dagger p>

    The convention is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec)
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        link_index = (link_indexa, link_indexb)
    rdm1a = rdm.make_rdm1_spin1('FCImake_rdm1a', fcivec, fcivec,
                                norb, nelec, link_index)
    rdm1b = rdm.make_rdm1_spin1('FCImake_rdm1b', fcivec, fcivec,
                                norb, nelec, link_index)
    return rdm1a, rdm1b

def make_rdm1(fcivec, norb, nelec, link_index=None):
    r'''Spin-traced one-particle density matrix

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention is based on McWeeney's book, Eq (5.4.20)
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    rdm1a, rdm1b = make_rdm1s(fcivec, norb, nelec, link_index)
    return rdm1a + rdm1b

def make_rdm12s(fcivec, norb, nelec, link_index=None, reorder=True):
    r'''Spin separated 1- and 2-particle density matrices.
    The return values include two lists, a list of 1-particle density matrices
    and a list of 2-particle density matrices.  The density matrices are:
    (alpha,alpha), (beta,beta) for 1-particle density matrices;
    (alpha,alpha,alpha,alpha), (alpha,alpha,beta,beta),
    (beta,beta,beta,beta) for 2-particle density matrices.

    1pdm[p,q] = :math:`\langle q^\dagger p\rangle`;
    2pdm[p,q,r,s] = :math:`\langle p^\dagger r^\dagger s q\rangle`.

    Energy should be computed as
    E = einsum('pq,qp', h1, 1pdm) + 1/2 * einsum('pqrs,pqrs', eri, 2pdm)
    where h1[p,q] = <p|h|q> and eri[p,q,r,s] = (pq|rs)
    '''
    dm1a, dm2aa = rdm.make_rdm12_spin1('FCIrdm12kern_a', fcivec, fcivec,
                                       norb, nelec, link_index, 1)
    dm1b, dm2bb = rdm.make_rdm12_spin1('FCIrdm12kern_b', fcivec, fcivec,
                                       norb, nelec, link_index, 1)
    _, dm2ab = rdm.make_rdm12_spin1('FCItdm12kern_ab', fcivec, fcivec,
                                    norb, nelec, link_index, 0)
    if reorder:
        dm1a, dm2aa = rdm.reorder_rdm(dm1a, dm2aa, inplace=True)
        dm1b, dm2bb = rdm.reorder_rdm(dm1b, dm2bb, inplace=True)
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

def make_rdm12(fcivec, norb, nelec, link_index=None, reorder=True):
    r'''Spin traced 1- and 2-particle density matrices.

    1pdm[p,q] = :math:`\langle q_\alpha^\dagger p_\alpha \rangle +
                       \langle q_\beta^\dagger  p_\beta \rangle`;
    2pdm[p,q,r,s] = :math:`\langle p_\alpha^\dagger r_\alpha^\dagger s_\alpha q_\alpha\rangle +
                           \langle p_\beta^\dagger  r_\alpha^\dagger s_\alpha q_\beta\rangle +
                           \langle p_\alpha^\dagger r_\beta^\dagger  s_\beta  q_\alpha\rangle +
                           \langle p_\beta^\dagger  r_\beta^\dagger  s_\beta  q_\beta\rangle`.

    Energy should be computed as
    E = einsum('pq,qp', h1, 1pdm) + 1/2 * einsum('pqrs,pqrs', eri, 2pdm)
    where h1[p,q] = <p|h|q> and eri[p,q,r,s] = (pq|rs)
    '''
    #(dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = \
    #        make_rdm12s(fcivec, norb, nelec, link_index, reorder)
    #return dm1a+dm1b, dm2aa+dm2ab+dm2ab.transpose(2,3,0,1)+dm2bb
    dm1, dm2 = rdm.make_rdm12_spin1('FCIrdm12kern_sf', fcivec, fcivec,
                                    norb, nelec, link_index, 1)
    if reorder:
        dm1, dm2 = rdm.reorder_rdm(dm1, dm2, inplace=True)
    return dm1, dm2


def _unpack_nelec(nelec):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    return neleca, nelecb