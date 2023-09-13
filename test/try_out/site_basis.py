import numpy as np
import scipy.linalg as sla


norb = 10
nocc = int(norb//2)
z = np.exp(1.j * 2 * np.pi * np.arange(norb)/norb)
Z = np.diag(z)

a = np.zeros((norb, nocc), dtype=np.complex128)
a[:nocc, :nocc] = np.eye(nocc)

b = np.copy(a) # excitation from a
b[2,2] = 0
b[6,2] = 1 

c = np.dot(Z, b)

ovlp = np.dot(a.conj().T, c)
print(sla.det(ovlp))

