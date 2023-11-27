import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

# set system
t = -1
nsite = 14
nelec = nsite
U = 4
pbc = True
spin = 0 # n_up - n_dn

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
driver.initialize_system(n_sites=nsite, n_elec=nelec, spin=spin)

# build Hamiltonian
# c - creation spin up, d - annihilation spin up
# C - creation spin dn, D - annihilation spin dn
ham_str = driver.expr_builder() 
ham_str.add_term("cd", np.array([[[i, i + 1], [i + 1, i]] for i in range(nsite - 1)]).flatten(), t)
ham_str.add_term("CD", np.array([[[i, i + 1], [i + 1, i]] for i in range(nsite - 1)]).flatten(), t)
if pbc:
    ham_str.add_term("cd", np.array([[nsite-1, 0], [0, nsite-1]]).flatten(), t)
    ham_str.add_term("CD", np.array([[nsite-1, 0], [0, nsite-1]]).flatten(), t)

ham_str.add_term("cdCD", np.array([[i, ] * 4 for i in range(nsite)]).flatten(), U)
ham_mpo = driver.get_mpo(ham_str.finalize(), iprint=0)

# run DMRG
def run_dmrg(driver, mpo, init_bond_dim=250, nroots=1):
    ket = driver.get_random_mps(tag="KET", bond_dim=init_bond_dim, nroots=nroots)
    bond_dims = [250] * 4 + [500] * 4 # schedule
    noises = [1e-4] * 4 + [1e-5] * 4 + [0]
    thrds = [1e-10] * 8
    return driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises,
        thrds=thrds, cutoff=0, iprint=1)

energies = run_dmrg(driver, ham_mpo)
print('DMRG energy = {}'.format(energies/nsite))
