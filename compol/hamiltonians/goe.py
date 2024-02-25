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
    def __init__(self, nsite, W1=1, W2=1, distrib="box", 
                 nelec=None, filling=0.5):
        '''
        Initializes the Spinless1D class.
        Args:
            nsite (int): Number of sites.
        Kwargs:
            W1 (float): Width of the distribution for h1e.
            W2 (float): Width of the distribution for h1e.
            distrib (str): Distribution type, can be "box" or "gaussian".
            nelec (int): Number of electrons.
            filling (float): Electron filling fraction (nelec / nsite). Used only when nelec is not provided.
        '''
        self.nsite = nsite 
        self.W1 = W1 
        self.W2 = W2
        self.distrib = distrib
        if nelec is None:
            nelec = int(nsite * filling + 1e-6) 
            f = nelec/nsite
            if abs(filling - nelec/nsite) > 1e-15:
                logging.warning("Changing filling from {:0.2f} to {:0.2f} to keep integer electron count!".format(filling, f))
        self.nelec = nelec
        self.print_info()

    def print_info(self):
        header = "1D GOE Hamiltonian with diagonal disorder."
        sys_info = " L = {:d}  |  Ne = {:d}  ".format(self.nsite, self.nelec)
        lh = len(header)
        print("#"*lh + "\n" + header + '\n' + "#"*lh)
        print(sys_info)
        print("-"*lh)
        print(" Distribution = {} | |   W1 = {}  |  W2 = {}".format(self.distrib, self.W1, self.W2))
        print("#"*lh)

    def gen_ham(self):
        """
        Generate the one-body and two-body Hamiltonian.
        """
        # create noise
        if self.distrib == "box":
            h1e = np.random.uniform(-self.W1, self.W1, (self.nsite, self.nsite))
            h2e = np.random.uniform(-self.W2, self.W2, (self.nsite**2, self.nsite**2))
        elif self.distrib == "gaussian":
            h1e = np.random.normal(0, self.W1, (self.nsite, self.nsite))
            h2e = np.random.normal(0, self.W2,  (self.nsite**2, self.nsite**2))
        else:
            raise ValueError("Distributions can only be 'box' or 'gaussian'!")

        
        for i in range(self.nsite):
            h2e[i, i] = 0
        h1e = (h1e + h1e.T) / 2.
        h2e = (h2e + h2e.T) / 2.
        h2e = h2e.reshape((self.nsite,)*4)
        eri = ao2mo.restore(8, h2e, self.nsite) # make h2e physical
        h2e_n = ao2mo.restore(1, eri, self.nsite)

        return h1e, h2e_n
    
    def gen_ham_uhf(self):
        h1e, h2e = self.gen_ham()
        return np.array([h1e, h1e*0]), np.array([h2e, h2e*0, h2e*0])

    def run_scf(self, mf_tol=1e-10, mf_niter=200, T=0, Tmin=1e-2):
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
        mf.verbose = 3
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
