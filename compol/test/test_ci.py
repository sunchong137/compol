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
from compol import civecs, hubbard
from scipy.special import comb
import time 


def test_gen_strs():
    norb = 4
    nelec = 2
    t1 = time.time()
    cistrs = civecs.gen_cistr(norb, nelec)
    t2 = time.time()
    print(t2-t1)
    ref = np.array([
        [1,1,0,0],
        [1,0,1,0],
        [0,1,1,0],
        [1,0,0,1],
        [0,1,0,1],
        [0,0,1,1]
    ])

    assert np.allclose(cistrs, ref)


def test_z_ci():

    #RHF 
    norb = 6
    nelec = 4
    len_ci = int(comb(norb, nelec))
    H, _ = hubbard.hubham_1d(norb, U=0) 
    _, mo = np.linalg.eigh(H)
    ci = np.random.rand(len_ci, len_ci)
    ci /= np.linalg.norm(ci)
    z = civecs.compol_fci_full(ci, norb, nelec, mo)
    assert np.abs(z) <= 1
    
    # UHF 
    norb = 6
    nelec = 6
    len_ci = int(comb(norb, nelec))
    H, _ = hubbard.hubham_1d(norb, U=0) 
    _, mo = np.linalg.eigh(H)
    mo = np.array([mo, mo])
    ci = np.random.rand(len_ci, len_ci)
    ci /= np.linalg.norm(ci)
    ci = np.zeros((len_ci, len_ci))
    ci[0,0] = 1
    z = civecs.compol_fci_full(ci, norb, nelec, mo)
    assert np.allclose(np.abs(z), 0)


def test_compare_full_site():

    norb = 4
    nelec = 2
    len_ci = int(comb(norb, nelec//2))
    H, _ = hubbard.hubham_1d(norb, U=4) 
    # _, mo = np.linalg.eigh(H) 
    mo = np.eye(norb)
    # np.random.seed(0)
    ci = np.random.rand(len_ci, len_ci)
    ci /= np.linalg.norm(ci)
    ci_strs = civecs.gen_cistr(norb, nelec//2)
    z = civecs.compol_fci_site(ci, norb, nelec, x0=0.0)
    z2 = civecs.compol_fci_full(ci, norb, nelec, mo, x0=0.0)

    assert np.allclose(z, z2)

def test_compol_prod():

    norb = 4
    # nelec = 4
    nelec = (2, 2)
    ci_strs = civecs.gen_cistr(norb, nelec[0])
    len_ci = len(ci_strs)
    # len_ci = int(comb(norb, nelec//2))
    # len_ci = int(comb(norb, nelec//2))
    x0 = 0
    mo_coeff = np.eye(norb)
 
    np.random.seed(0)
    ci = np.random.rand(len_ci, len_ci)
    ci = ci + ci.T
    ci /= np.linalg.norm(ci)
    # ci = np.zeros_like(ci)
    # ci[0,0] = 1
    z = civecs.compol_fci_prod(ci, norb, nelec, x0=x0)
    z2 = civecs.compol_fci_site(ci, norb, nelec, x0=x0)
    z3 = civecs.compol_fci_full(ci, norb, nelec, mo_coeff, x0)
    print(z)
    print(z2)
    print(z3)

test_compol_prod()