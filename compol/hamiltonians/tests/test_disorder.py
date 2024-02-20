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
from pyscf import gto, scf, ao2mo, fci

def test_ham():
    nsite = 6
    V = 1 
    tprime = -2
    W = 3 
    pbc = False
    h1e, h2e = disorder_ham.ham_disorder_1d(nsite, V, W, tprime, pbc=pbc, dist='box')
    print(h2e)

def test_mf():
    nsite = 6
    V = 1 
    tprime = -2
    W = 3 
    pbc = False
    filling=1.0

    mf = disorder_ham.mf_disorder(nsite, V, W, tprime, pbc=pbc, dist='box', filling=filling)
    print(mf.e_tot)

test_mf()