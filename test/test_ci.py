import numpy as np
import sys
sys.path.append("../")
import civecs, hubbard
from pyscf import fci


def compol_fci():
    n = 10
    U = 0
    h1 = np.zeros((n,n))
    for i in range(n-1):
        h1[i,i+1] = h1[i+1,i] = -1.0
    h1[n-1,0] = h1[0,n-1] = -1.0  # PBC
    eri = np.zeros((n,n,n,n))
    for i in range(n):
        eri[i,i,i,i] = U
    myci = fci.direct_spin1
    e, c = myci.kernel(h1, eri, n, n)
    z = civecs.compol_fci(c, n, n, x0=-0)
    print(z)
    
    # compare to UHF
    
compol_fci()