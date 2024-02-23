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
Evaluate complex polarization based on Slater determinants at finite temperature.
1D model system with site basis only, 
'''
from compol import slater
import numpy as np

Pi = np.pi
def gen_zmat_site_ft(L, x0):
    '''
    Generate the matrix for Z operator in the site basis for MO coeffs.
    '''
    pos = np.arange(L) + x0 
    Z = np.eye(L*2, dtype=np.complex128)
    Z[:L, :L] = np.diag(np.exp(2.j * Pi * pos / L))
    return Z
    