import numpy as np
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.algebra.mpe import MPE

nsite = 6
U = 4
# # generate h1e and g2e for Hubbard model
# h1e, g2e = hubbard.hubham_1d(nsite, U, pbc=True, noisy=False, max_w=1.0, spin=0)

def build_hubbard_mpo(nsite, nelec, U, t=-1, pbc=False, cutoff=1E-9):
    fcidump = FCIDUMP(pg='c1', n_sites=nsite, n_elec=nelec, twos=0, ipg=0, orb_sym=[0] * nsite)
    hamil = Hamiltonian(fcidump, flat=True)

    def generate_terms(n_sites, c, d):
        # hopping 
        for i in range(0, n_sites-1):
            for s in [0, 1]:
                yield t * c[i, s] * d[i+1, s]
                yield t * c[i+1, s] * d[i, s]
        for i in range(n_sites):
            yield U * (c[i, 0] * c[i, 1] * d[i, 1] * d[i, 0])
        if pbc:
            for s in [0, 1]:
                yield t * c[0, s] * d[n_sites-1, s]
                yield t * c[n_sites-1, s] * c[0, s]

    return hamil, hamil.build_mpo(generate_terms, cutoff=cutoff).to_sparse()

hamil, ham_mpo = build_hubbard_mpo(nsite, nsite, U)

# constrruct initial MPS
bond_dim = 100
mps = hamil.build_mps(bond_dim)
# print(np.dot(mps, ham_mpo @ mps))
dmrg = MPE(mps, ham_mpo, mps).dmrg(bdims=[bond_dim], noises=[1E-6, 0], dav_thrds=[1E-3], iprint=2, n_sweeps=10)
ener = dmrg.energies[-1]
print(ener/nsite)

# build a random MPO
def build_nn_mpo(nsite, nelec, cutoff=1e-9):
    fcidump = FCIDUMP(pg='c1', n_sites=nsite, n_elec=nelec, twos=0, ipg=0, orb_sym=[0] * nsite)
    hamil = Hamiltonian(fcidump, flat=True)

    def generate_terms(nsites, c, d):
        yield c[0, 0] * d[0, 0]
    return hamil, hamil.build_mpo(generate_terms, cutoff=cutoff).to_sparse()

nn, nn_mpo = build_nn_mpo(nsite, nsite)

nmps = nn_mpo @ mps

n = np.dot(nmps, mps)
print(n)