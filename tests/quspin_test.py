from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt

n = int(input("number of bath qubits : "))
A = np.ones(n)

# compute Hilbert space basis
basis = spin_basis_1d(L = n+1)

# compute site-coupling lists
z_term = [[-0.5, 0]]
couple_term = [[A[i], 0, i+1] for i in range(n)]
x_term = [[1.0, 0]]

#operator string lists
static_d = [['z', z_term], ['zz', couple_term],
            ['xx', couple_term],['yy', couple_term]]
static_c = [['x', x_term]]

#The drifting Hamiltonian
H_da = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
#The control Hamiltonian
H_ca = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)

#QAOA Hamiltonians
H0a = H_da.toarray() + 2.0 * H_ca.toarray()
H1a = H_da.toarray() - 2.0 * H_ca.toarray()
# print(np.real(H0a))
# print(np.imag(H0a))

import Wastes.CentralSpin as CentralSpin

sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
sigma_y = 1.0j * np.array([[0.0, - 1.0], [1.0, 0.0]])

H_db = CentralSpin.Hamiltonian(-0.5*sigma_z, A)
H_cb = np.kron(sigma_x, np.identity(2**n))
#QAOA Hamiltonians
H0b = H_db + 2.0 * H_cb
H1b = H_db - 2.0 * H_cb
# print(np.real(H0b))
# print(np.imag(H0b))

plt.matshow(np.real(H0a-H0b))
plt.show()
print(np.real(H0a-H0b))
plt.matshow(np.imag(H0a-H0b))
plt.show()
plt.matshow(np.real(H1a-H1b))
plt.show()
plt.matshow(np.imag(H1a-H1b))
plt.show()
