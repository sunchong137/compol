import numpy as np
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.algebra.mpe import MPE


def hubbard1d_dmrg(nsite, U, nelec=None, filling=1.0, pbc=False,
                   init_bdim=50, max_bdim=200, nsweeps=8, cutoff=1e-8,
                   max_noise=1e-5):
    '''
    Run DMRG on the 1D Hubbard model.
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



# # build a random MPO
# def build_nn_mpo(nsite, nelec, cutoff=1e-9):
#     fcidump = FCIDUMP(pg='c1', n_sites=nsite, n_elec=nelec, twos=0, ipg=0, orb_sym=[0] * nsite)
#     hamil = Hamiltonian(fcidump, flat=True)

#     def generate_terms(nsites, c, d):
#         yield c[0, 0] * d[0, 0]
#     return hamil, hamil.build_mpo(generate_terms, cutoff=cutoff).to_sparse()

# nn, nn_mpo = build_nn_mpo(nsite, nsite)

# nmps = nn_mpo @ mps

# n = np.dot(nmps, mps)
# print(n)