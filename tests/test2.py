#Testing the scenario of XmonTLS
import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
n=3

g = np.random.uniform(0.1, 1, n)
g /= 5.5*1000 #Scale to desired strength
print("g:")
print(g)

Delta = np.linspace(1.0,1.0+0.1*(n-1), num=n) #TLS frequencies
Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency
print("Delta:", Delta)

# compute Hilbert space basis
basis = spin_basis_1d(L = n+1)

# compute site-coupling lists
z_term = [[-Delta[i]/2, i] for i in range(n+1)]
print("z_term:")
print(z_term)
couple_term = [[g[i]/2, 0, i+1] for i in range(n)]
print("couple_term:")
print(couple_term)
x_term = [[1.0, 0]]

#operator string lists
static_d = [['z', z_term], 
            ['-+', couple_term], ['+-', couple_term]]
static_c = [['x', x_term]]

#The drifting Hamiltonian
H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
print(H_d)
#The control Hamiltonian
H_c = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)
print(H_c)
#QAOA Hamiltonians
H0 = H_d.toarray() + 2.0 * H_c.toarray()
print(H0)
H1 = H_d.toarray() - 2.0 * H_c.toarray()
print(H1)