import numpy as np
from pyscf import ao2mo

def run_stab_mf(mf):
    '''
    Run mf with stabalizer.
    '''
    mf.kernel()
    mo1 = mf.stability()[0]
    init = mf.make_rdm1(mo1, mf.mo_occ)                                                 
    mf.kernel(init)

def rotate_ham(mf):
    '''
    Rotate the Hamiltonian from atomic orbitals to molecular orbitals.
    '''
    h1e = mf.get_hcore()
    norb = mf.mol.nao
    eri = mf._eri
    mo_coeff = mf.mo_coeff
    # aaaa, aabb, bbbb
    Ca, Cb = mo_coeff[0], mo_coeff[1]
    aaaa = (Ca,)*4
    bbbb = (Cb,)*4
    aabb = (Ca, Ca, Cb, Cb)

    h1_mo = np.array([Ca.T @ h1e @ Ca, Cb.T @ h1e @ Cb])
    h2e_aaaa = ao2mo.incore.general(eri, aaaa, compact=False).reshape(norb, norb, norb, norb)
    h2e_bbbb = ao2mo.incore.general(eri, bbbb, compact=False).reshape(norb, norb, norb, norb)
    h2e_aabb = ao2mo.incore.general(eri, aabb, compact=False).reshape(norb, norb, norb, norb)
    h2_mo = np.array([h2e_aaaa, h2e_aabb, h2e_bbbb])

    return h1_mo, h2_mo