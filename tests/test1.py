#Check if the dipole-dipole interaction is implemented correctly
import numpy as np
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
sigma_y = 1.0j * np.array([[0.0, - 1.0], [1.0, 0.0]])
sigma_p = 2*np.array([[0.0, 1.0], [0.0, 0.0]])
sigma_m = 2*np.array([[0.0, 0.0], [1.0, 0.0]])

t1 = np.kron(sigma_x,sigma_x) + np.kron(sigma_y,sigma_y)
print(t1)

t2 = np.kron(sigma_p,sigma_m) + np.kron(sigma_m,sigma_p)
print(t2)

from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

# compute Hilbert space basis
basis = spin_basis_1d(2)

# compute site-coupling lists
couple_term = [[1, 0, 1] ]

#operator string lists
static_d = [['+-', couple_term], ['-+', couple_term]]

#The drifting Hamiltonian
H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
#The control Hamiltonian

#QAOA Hamiltonians
t3 = H_d.toarray()
print(t3)