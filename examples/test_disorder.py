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
from compol import helpers
from compol.hamiltonians import disorder_ham
from compol.solvers import fci_uhf
from pyscf.fci import direct_uhf
from pyscf import fci, ao2mo 


def test_all():
    nsite = 10
    V = 2
    tprime = -1
    W = 4
    pbc = False
    obj = disorder_ham.spinless1d(nsite, V, W, tprime, pbc, 'box')
    mf = obj.gen_scf()
    h1e = mf.get_hcore()
    eri = mf._eri
    nelec = mf.nelec
    mo = mf.mo_coeff
    # m_occ = np.zeros(nsite)
    # m_occ[:nelec[0]] = 1
    # np.random.shuffle(m_occ)
    # rdm1_mf = mf.make_rdm1(mo, [m_occ, m_occ*0])[0]
    # ew, ev = np.linalg.eigh(rdm1_mf)
    # print(ew)
    # exit()
    # mo_e = mf.mo_energy
    # print(mo_e)
    # veff = mf.get_veff()
    # print(veff)
    # exit()
    e_hf = mf.e_tot

    h1e_mo, h2e_mo = helpers.ao2mo_spinless(h1e, eri, mo)
    e, v = fci_uhf.kernel(h1e_mo, h2e_mo, nsite, nelec, target_e=e_hf/2)
    # myci = fci.FCI(mf)
    # e2, v2 = myci.kernel()
    # print(v)
    rdm1, rdm2 = fci_uhf.make_rdm12(v, nsite, nelec)
    ew, ev = np.linalg.eigh(rdm1)
    print(ew)


test_all()