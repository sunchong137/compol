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
import logging
from pyscf import gto, scf, ao2mo
from pyscf.scf.addons import smearing_

class spinless1d(object):
    '''
    Represents a one-dimensional spinless Hamiltonian with diagonal disorder.
    The Hamiltonian is given by:
    ```math
    H = -\sum_{\langle i, j\rangle} (a^\dagger_i a_j + h.c.)
        -t^' \sum_{\llangle i, j\rrangle} (a^\dagger_i a_j + h.c.)
        + \sum_i w_i a^\dagger_i a_i 
        + V\sum_{\langle i, j\rangle} n_i n_j
    FCI can take 18 sites, maybe more. 
    '''
    def __init__(self, nsite, V, W=1, tprime=0, pbc=False, distrib="box", 
                 nelec=None, filling=0.5):
        '''
        Initializes the Spinless1D class.
        Args:
            nsite (int): Number of sites.
            V (float): Nearest-neighbor two-body interaction strength.

        Kwargs:
            W (float): Width of the distribution.
            tprime (float): Next-nearest-neighbor hopping amplitude.
            pbc (bool): If True, periodic boundary condition is used.
            distrib (str): Distribution type, can be "box" or "gaussian".
            nelec (int): Number of electrons.
            filling (float): Electron filling fraction (nelec / nsite). Used only when nelec is not provided.
        '''
        self.nsite = nsite 
        self.V = V
        self.W = W 
        self.tprime = tprime
        self.pbc = pbc 
        self.distrib = distrib
        if nelec is None:
            nelec = int(nsite * filling + 1e-6) 
            f = nelec/nsite
            if abs(filling - nelec/nsite) > 1e-15:
                logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer electron count!".format(filling, f))
        self.nelec = nelec
        self.print_info()

    def print_info(self):
        header = "1D spinless Hamiltonian with diagonal disorder."
        sys_info = " L = {:d}  |  Ne = {:d}  |  V = {:0.1f}  |  PBC = {}".format(self.nsite, self.nelec, self.V, self.pbc)
        lh = len(header)
        print("#"*lh + "\n" + header + '\n' + "#"*lh)
        print(sys_info)
        print("-"*lh)
        print(" tprime = {:0.2f}  ".format(self.tprime))
        print(" Distribution = {} | Width = {:0.2f}".format(self.distrib, self.W))
        print("#"*lh)

    def gen_ham(self):
        """
        Generate the one-body and two-body Hamiltonian.
        """
        h1e = np.zeros((self.nsite, self.nsite))
        h2e = np.zeros((self.nsite,) * 4)

        # create noise
        if self.distrib == "box":
            noise = np.random.uniform(-self.W, self.W, self.nsite)
        elif self.distrib == "gaussian":
            noise = np.random.normal(0, self.W, self.nsite)
        else:
            raise ValueError("Distributions can only be 'box' or 'gaussian'!")

        # 1-body term
        for i in range(self.nsite - 2):
            h1e[i, i+1] = h1e[i+1, i] = -1.
            h1e[i, i+2] = h1e[i+2, i] = self.tprime
        h1e[-2, -1] = h1e[-1, -2] = -1.

        # 2-body term
        for i in range(self.nsite - 1):
            h2e[i, i, i+1, i+1] = h2e[i+1, i+1, i, i] = self.V / 2.  # making h2e symmetric

        if self.pbc:
            h1e[0, -1] = h1e[-1, 0] = -1.
            h1e[0, -2] = h1e[-2, 0] = self.tprime
            h2e[0, -1] = h2e[-1, 0] = self.V / 2.
        h1e += np.diag(noise)
        return h1e, h2e
    
    def gen_ham_uhf(self):
        h1e, h2e = self.gen_ham()
        return np.array([h1e, h1e*0]), np.array([h2e, h2e*0, h2e*0])

    def run_scf(self, mf_tol=1e-10, mf_niter=100, T=0, Tmin=1e-2):
        """
        Generate the PySCF meanfield object.
        """
        mol = gto.M()
        mol.nelectron = self.nelec
        mol.tot_electrons = lambda *args: np.sum(self.nelec)
        mol.nao = self.nsite
        mol.spin = self.nelec # spinless
        h1e, eri = self.gen_ham()
        mol.build()
        mf = scf.UHF(mol)
        # add temperature 
        if T > Tmin:
            print("Running finite-temperature SCF!")
            beta = 1/T 
            mf = smearing_(mf, beta, 'fermi', fix_spin=False)
        mf.get_hcore = lambda *args: h1e
        mf.get_ovlp = lambda *args: np.eye(self.nsite)
        mf._eri = ao2mo.restore(1, eri, self.nsite)
        mol.incore_anyway = True
        mf.conv_tol = mf_tol
        mf.max_cycle = mf_niter


        mf.kernel()
        return mf

    def gen_ham_full():
        pass 

# def ham_disorder_1d(nsite, V, W=1, tprime=0, pbc=False, dist="box"):
#     '''
#     Generate the Hamiltonian for the disorder model. 
#     Two-body Hamiltonian has the form a_i^\dag a_i a^\dag_j a_j
#     Args:
#         nsite: int, number of sites.
#         V: float, nearest-neighbor two-body interaction strength.
#         W: float, The width of the distribution
#         tprime: float, next--nearest-neighbor hopping amplitude
#     '''
#     h1e = np.zeros((nsite, nsite))
#     h2e = np.zeros((nsite,)*4)

#     # create noise
#     if dist == "box":
#         noise = np.random.uniform(-W, W, nsite)
#     elif dist == "gaussian":
#         noise = np.random.normal(0, W, nsite)
#     else:
#         raise ValueError("Distributions can only be 'box' or 'gaussian'!")

#     # 1-body term
#     for i in range(nsite-2):
#         h1e[i, i+1] = h1e[i+1, i] = -1.
#         h1e[i, i+2] = h1e[i+2, i] = tprime
#     h1e[-2, -1] = h1e[-1, -2] = -1.
#     h1e += np.diag(noise)

#     # 2-body term
#     for i in range(nsite-1):
#         h2e[i, i, i+1, i+1] = h2e[i+1, i+1, i, i] = V/2. # making h2e symmetric 

#     if pbc:
#         h1e[0, -1] = h1e[-1, 0] = -1. 
#         h1e[0, -2] = h1e[-2, 0] = tprime    
#         h2e[0, -1] = h2e[-1, 0] = V/2.
    
#     return h1e, h2e 


# def mf_disorder(nsite, V, W=1, tprime=0, pbc=False, dist="box", 
#                 filling=1.0, nelec=None, mf_tol=1e-10, mf_niter=100):
#     '''
#     Generate the pyscf meanfield object 
#     '''
#     mol = gto.M()
    
#     if nelec is None:
#         filling /= 2 # spinless 
#         nelec = int(nsite * filling + 1e-10)
#         if abs(nelec - nsite * filling) > 1e-2:
#             logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer number of electrons!".format(filling, nelec/nsite))
    

#     mol.nelectron = nelec
#     mol.nao = nsite
#     mol.spin = nelec
#     h1e, eri = ham_disorder_1d(nsite, V, W=W, tprime=tprime, pbc=pbc, dist=dist)
    
#     mf = scf.UHF(mol)
#     mf.get_hcore = lambda *args: h1e 
#     mf.get_ovlp = lambda *args: np.eye(nsite)
#     mf._eri = ao2mo.restore(8, eri, nsite)
#     mol.incore_anyway = True
#     mf.conv_tol = mf_tol
#     mf.max_cycle = mf_niter
#     mf.kernel()
#     return mf

