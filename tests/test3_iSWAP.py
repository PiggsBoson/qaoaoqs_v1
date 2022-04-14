#Testing the scenario of iSWAP
import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

n1 = 1#number of bath qubits coupled to qubit 1
n2 = 2 #number of bath qubits coupled to qubit 2

A = np.random.uniform(0.5, 5, n1+n2)
A /= 1000 #Scale to desired strength
print("A:")
print(A)

g = 2.5/1000 #qubit-qubit coupling constant
print("g:",g)
# compute Hilbert space basis
basis = spin_basis_1d(L = n1+n2+2)

# compute site-coupling lists
couple_term_1 = [[A[i]/2, 0, i+2] for i in range(n1)]
print("couple_term_1:")
print(couple_term_1)
couple_term_2 = [[A[i+n1]/2, 1, i+2+n1] for i in range(n2)]
print("couple_term_2:")
print(couple_term_2)
x_term = [[1.0, i] for i in range(2)]
print("x_term:")
print(x_term)
entangle_term = [[g/2, 0, 1]]
print("entangle_term:")
print(entangle_term)

#operator string lists
static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
            ['-+', couple_term_2], ['+-', couple_term_2]]
control_a = [['x', x_term]]
control_b = [['-+', entangle_term], ['+-', entangle_term]]

#The drifting Hamiltonian
H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
print(H_d)
#The control Hamiltonians
H_ca = hamiltonian(control_a, [], basis=basis, dtype=np.complex128)
print(H_ca)
H_cb = hamiltonian(control_b, [], basis=basis, dtype=np.complex128)
print(H_cb)

#QAOA Hamiltonians
H0 = H_d.toarray() + H_ca.toarray()
H1 = H_d.toarray() + H_cb.toarray()
