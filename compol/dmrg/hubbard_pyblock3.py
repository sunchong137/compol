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
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.algebra.mpe import MPE

Pi = np.pi

def hubbard1d_dmrg(nsite, U, nelec=None, filling=1.0, pbc=False,
                   init_bdim=50, max_bdim=200, nsweeps=8, cutoff=1e-8,
                   max_noise=1e-5):
    '''
    Run DMRG on the 1D Hubbard model.
    Args:
        nsite: int, number of sites.
        U: float, on-site Coulomb interaction amplitude, unit: hopping amplitude t.
    Kwargs:
        nelec: int or tuple, if None, derived from filling.
        filling: float, nelec/nsite. filling=1 means half-filling.
        pbc: bool, if True, hopping between the first and last sites are included in the Hamiltonian.
        init_bdim: int, initial bond dimension.
        max_bdim: int, maximum bond dimension.
        nsweeps: int, number of DMRG sweep.
        cutoff: float, SVD singular value cutoff.
        max_noise: float, starting noise, decay to 0.
    Returns:
        energy: float, the DMRG ground state energy.
        mps: the DMRG ground state MPS.
        mpo: the Hamiltonian
    '''
    if nelec is None:
        nelec = int(nsite * filling + 1e-10)
        if abs(nelec/nsite - filling) > 1e-5:
            print("WARNING: The filling is changed to {:1.2f}".format(nelec/nsite))
        spin = 0
    else:
        try:
            neleca, nelecb = nelec
            spin = abs(neleca - nelecb)
            nelec = neleca + nelecb 
        except:
            spin = 0  

    # construct Hubbard MPO
    fcidump = FCIDUMP(pg='c1', n_sites=nsite, n_elec=nelec, twos=spin, ipg=0, orb_sym=[0] * nsite)
    hamil = Hamiltonian(fcidump, flat=True) # flat=True to use C++
    def generate_terms(n, c, d):
        # hopping 
        for i in range(0, n-1):
            for s in [0, 1]:
                yield -1 * c[i, s] * d[i+1, s]
                yield -1 * c[i+1, s] * d[i, s]
        for i in range(n):
            yield U * (c[i, 0] * c[i, 1] * d[i, 1] * d[i, 0])
        if pbc:
            for s in [0, 1]:
                yield -1 * c[0, s] * d[n-1, s]
                yield -1 * c[n-1, s] * c[0, s]
    ham_mpo = hamil.build_mpo(generate_terms, cutoff=cutoff).to_sparse()

    # initialize MPS
    mps = hamil.build_mps(init_bdim)
    # Schedule, using the linearly growing bond dimonsion
    bdims = list(np.linspace(init_bdim, max_bdim, nsweeps//2, endpoint=True, dtype=int))
    if max_noise < 1e-16:
        noises = [0.0]
    else:
        noises = list(np.logspace(np.log(max_noise), -16, nsweeps//2, endpoint=True)) + [0.0]

    # run DMRG
    dmrg = MPE(mps, ham_mpo, mps).dmrg(bdims=bdims, noises=noises, dav_thrds=None, iprint=1, n_sweeps=nsweeps)
    energy = dmrg.energies[-1]
    print("Total energy: {:2.6f}; Energy per site: {:2.6f}".format(energy, energy/nsite))
    return energy, mps, ham_mpo


def spinless_dmrg():
    pass 

def compol_prod(mps, nsite, nelec, x0=0.0, max_bdim=400, cutoff=1e-8, tol=1e-8):

    ket_mps = np.copy(mps)
    fcidump = FCIDUMP(pg='c1', n_sites=nsite, n_elec=nelec, twos=0, ipg=0, orb_sym=[0] * nsite)
    operator = Hamiltonian(fcidump, flat=True)
    for site in range(nsite):
        coeff = np.exp(2.j*Pi*(site-x0)/nsite)-1.0
        # spin up
        def generate_terms(nsites, c, d): # up term
            yield c[site, 0] * d[site, 0]
        mpo = operator.build_mpo(generate_terms, cutoff=cutoff).to_sparse()
        ket_mps += mpo @ ket_mps * coeff    
        ket_mps, c_err = ket_mps.compress(max_bond_dim=max_bdim, cutoff=cutoff)
        if c_err > tol:
            print("WARNING: compression error is {:0.4E}, greater than {:0.4E}".format(c_err, tol))
        # spin-down
        def generate_terms(nsites, c, d): # up term
            yield c[site, 1] * d[site, 1]
        mpo = operator.build_mpo(generate_terms, cutoff=cutoff).to_sparse()
        ket_mps += mpo @ ket_mps * coeff
        ket_mps, c_err = ket_mps.compress(max_bond_dim=max_bdim, cutoff=cutoff)
        if c_err > tol:
            print("WARNING: compression error is {:0.4E}, greater than {:0.4E}".format(c_err, tol))

    Z = np.dot(mps.conj(), ket_mps)
    return Z #np.linalg.norm(Z)
