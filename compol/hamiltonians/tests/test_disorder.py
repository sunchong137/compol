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
    nsite = 12
    V = 3
    tprime = -2
    W = 3
    pbc = False
    obj = disorder_ham.spinless1d(nsite, V, W, tprime, pbc, 'box')
    mf = obj.gen_scf()
    h1e = mf.get_hcore()
    eri = mf._eri
    nelec = mf.nelec
    mo = mf.mo_coeff
    e_hf = mf.e_tot
    # h1e_mo = np.array(
    #     [mo[0].T @ h1e @ mo[0], mo[1].T @ h1e @ mo[1]]
    # )
    # print(h1e_mo[0][0])
    # h2aa = ao2mo.kernel(eri, (mo[0], mo[0], mo[0], mo[0]))
    # h2ab = ao2mo.kernel(eri, (mo[0], mo[0], mo[1], mo[1]))
    # h2bb = ao2mo.kernel(eri, (mo[1], mo[1], mo[1], mo[1]))
    # h2e_mo = np.array([h2aa, h2ab, h2bb])
    h1e_mo, h2e_mo = helpers.ao2mo_spinless(h1e, eri, mo)
    e, v = fci_uhf.kernel(h1e_mo, h2e_mo, nsite, nelec, target_e=e_hf/2)
    # myci = fci.FCI(mf)
    # e2, v2 = myci.kernel()
    print(v)
    # print(e, e2)


test_all()