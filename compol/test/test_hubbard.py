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
from compol import hubbard, helpers
from pyscf import gto, scf, ao2mo, fci

def test_mf_rhf():
    norb = 6
    U = 4
    spin = 0
    mymf = hubbard.hubbard_mf(norb, U, spin=spin)
    e = mymf.energy_elec()[0]
    mo = mymf.mo_coeff 

    # pyscf example
    mymol = gto.M() 
    n = 6
    mymol.nelectron = n

    mf = scf.RHF(mymol)
    h1 = np.zeros((n,n))
    for i in range(n-1):
        h1[i,i+1] = h1[i+1,i] = -1.0
    h1[n-1,0] = h1[0,n-1] = -1.0  # PBC
    eri = np.zeros((n,n,n,n))
    for i in range(n):
        eri[i,i,i,i] = 4.0

    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(n)
    # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
    # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
    mf._eri = ao2mo.restore(8, eri, n)

    mf.kernel()

    e_hf = mf.energy_elec()[0]
    mo_coeff = mf.mo_coeff

    assert np.allclose(e, e_hf)
    assert np.allclose(mo, mo_coeff)
    
def test_hubbard_2d():
    nx = 3; ny = 2
    U = 4
    h1e, h2e = hubbard.hamhub_2d(nx, ny, U, pbc=True)
    h1e_ref = np.array([
        [0, -1, -1, -1, 0, 0],
        [-1, 0, -1, 0, -1, 0],
        [-1, -1, 0, 0, 0, -1],
        [-1, 0, 0, 0, -1, -1],
        [0, -1, 0, -1, 0, -1],
        [0, 0, -1, -1, -1, 0]
    ])
    assert np.allclose(h1e, h1e_ref)


def test_uhf():
    norb = 18
    U = 8
    spin = 1
    mymf = hubbard.hubbard_mf(norb, U, spin=spin)
    e = mymf.energy_elec()[0]
    mo = mymf.mo_coeff 

    # pyscf example
    mymol = gto.M() 
    mymol.nelectron = norb

    mf = scf.UHF(mymol)
    h1 = np.zeros((norb, norb))
    for i in range(norb - 1):
        h1[i,i+1] = h1[i+1,i] = -1.0
    h1[norb-1,0] = h1[0,norb-1] = -1.0  # PBC
    eri = np.zeros((norb,)*4)
    for i in range(norb):
        eri[i,i,i,i] = U

    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(norb)
    # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
    # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
    mf._eri = ao2mo.restore(8, eri, norb)

    helpers.run_stab_mf(mf)
    e_hf = mf.energy_elec()[0]
    mo_coeff = mf.mo_coeff

    assert np.allclose(e, e_hf)

def test_fci():
    bethe = -0.573729
    norb = 6
    U = 4
    pbc = True
    spin = 0
    # t1 = time.time()
    mymf = hubbard.hubbard_mf(norb, U, spin=spin, pbc=pbc)
    e_fci, ci = hubbard.hubbard_fci_from_mf(mymf)
    # t2 = time.time()
    e_fci2, ci2 = hubbard.hubbard_fci(norb, U, pbc=pbc)
    # t3 = time.time()
    assert np.allclose(e_fci, e_fci2)
    # print(t2-t1, t3-t2)
    # exit()
    e_per_site = e_fci / norb
    assert abs(e_per_site - bethe) < 1e-1

def test_filling():
    '''
    Test different fillings.
    '''
    norb = 18
    U = 8
    spin = 1
    mymf = hubbard.hubbard_mf(norb, U, spin=spin, filling=0.6)
    e = mymf.energy_elec()[0]
    
def test_disorder():
    pass

def test_spinless():
    nsite = 10
    V = 4
    mf = hubbard.hubbard_spinless_mf(nsite, V, nelec=None, pbc=False, filling=0.8)
    rdm1 = mf.make_rdm1()


def test_spinless_fci():
    norb = 10
    U = 4
    spin = 1 
    nelec = 5

    # first do a mean-field calculation
    mymf = hubbard.hubbard_spinless_mf(norb, U, nelec=nelec, pbc=True)
    rdm1 = mymf.make_rdm1()
    mo_coeff = mymf.mo_coeff

    h1_mo, h2_mo = helpers.rotate_ham_spinless(mymf)

    # FCI
    cisolver = fci.direct_uhf.FCI()
    e_fci, civec = cisolver.kernel(h1_mo, h2_mo, norb, (nelec, 0))

    # # compare to pyscf 
    h1e, eri = hubbard.hubham_spinless_1d(norb, U, pbc=True)
    h1_0 = np.zeros_like(h1e)
    h2_0 = np.zeros_like(eri)

    cisolver = fci.direct_uhf
    h1_uhf = (h1e, h1_0)
    h2_uhf = (eri, h2_0, h2_0)
    # h1_uhf = (h1e, h1e)
    # h2_uhf = (eri, eri, eri)
    e_fci_ref, c = cisolver.kernel(h1_uhf, h2_uhf, norb, (nelec, 0))
    assert np.allclose(e_fci, e_fci_ref)
