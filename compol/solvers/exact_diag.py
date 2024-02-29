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

from scipy import special
from pyscf.fci import direct_uhf

def fci_ham_pspace(h1e, eri, norb, nelec, max_np=1e4):
    '''
    Construct the full Hamiltonian using the pspace method.
    maximum: 8 orbitals.
    '''
    # get the Hamiltonian matrix
    try:
        neleca, nelecb = nelec 
    except:
        neleca = nelec//2
        nelecb = nelec - neleca 

    num_np = int(special.comb(norb, neleca) * special.comb(norb, nelecb) + 1e-5)
    if num_np > max_np:
        print(f"Warning: Exceeded memeory. P space is truncated to {max_np} points.")
        num_np = max_np    
    H_fci = direct_uhf.pspace(h1e, eri, norb, nelec, np=num_np)[1]
    return H_fci 