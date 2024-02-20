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

def ham_disorder_1d(nsite, V, W=1, tprime=0, pbc=False, dist="box"):
    '''
    Generate the Hamiltonian for the disorder model. 
    Two-body Hamiltonian has the form a_i^\dag a_i a^\dag_j a_j
    Args:
        nsite: int, number of sites.
        V: float, nearest-neighbor two-body interaction strength.
        W: float, The width of the distribution
        tprime: float, next--nearest-neighbor hopping amplitude
    '''
    h1e = np.zeros((nsite, nsite))
    h2e = np.zeros((nsite,)*4)

    # create noise
    if dist == "box":
        noise = np.random.uniform(-W, W, nsite)
    elif dist == "gaussian":
        noise = np.random.normal(0, W, nsite)
    else:
        raise ValueError("Distributions can only be 'box' or 'gaussian'!")

    # 1-body term
    for i in range(nsite-2):
        h1e[i, i+1] = h1e[i+1, i] = -1.
        h1e[i, i+2] = h1e[i+2, i] = tprime
    h1e[-2, -1] = h1e[-1, -2] = -1.
    h1e += np.diag(noise)

    # 2-body term
    for i in range(nsite-1):
        h2e[i, i, i+1, i+1] = h2e[i+1, i+1, i, i] = V/2. # making h2e symmetric 

    if pbc:
        h1e[0, -1] = h1e[-1, 0] = -1. 
        h1e[0, -2] = h1e[-2, 0] = tprime    
        h2e[0, -1] = h2e[-1, 0] = V/2.
    
    return h1e, h2e 

def mf_disorder():
    '''
    Generate the pyscf meanfield object 
    '''
    pass