import numpy as np
import scipy.linalg as sla


z = np.exp(1.j * 2 * np.pi * np.arange(6)/6)
Z = np.diag(z)

a = np.zeros((6, 4), dtype=np.complex128)
a[:4, :4] = np.eye(4)

b = np.copy(a) # excitation from a
b[2,2] = 0
b[5,2] = 1 

c = np.dot(Z, b)

ovlp = np.dot(b.conj().T, c)
print(sla.det(ovlp))

