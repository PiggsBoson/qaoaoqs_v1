#this will enable us to import modules from a parent directory. But it's generally a bad practice. It's better to build a pacakge!
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import Dynamics as dyn

import matplotlib.pyplot as plt
import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
import time
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype = np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype = np.complex128)
Y = 1.0j * np.array([[0.0, -1.0], [1.0, 0.0]],dtype = np.complex128)
pop_0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype = np.complex128)
coh_01 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype = np.complex128)

n_max = 15
EQCOUPLINGS = True
t_f = 2
T = np.linspace(0,t_f,200)
OBSERVABLE = X
cp_type = 'iso'
b_init = 'zero'
save = True
fname = "zeroT15"

def evolveNmeasure(rho0, H, T, OBSERVABLE):
	"""Get the fidelity with a given initial state in the vector form and a target state vector only for the system.

	Arguments:
		protocol -- The alpha's and beta's for a given protocol

	Returns:
		fildeity -- scalar between 0 and 1
	"""
	simulator = dyn.Dyna(rho0, H)
	rho1 = simulator.simulate_closed_rho(T)

	#The following is partial trace according to Peijun Zhu https://www.peijun.me/reduced-density-matrix-and-partial-trace.html
	#checked: this implementation works!
	N = len(rho1)//2 #size of bath space
	u_tensor=rho1.reshape([2, N, 2, N])
	u_a_mtx = np.trace(u_tensor, axis1=1, axis2=3)
	result = np.trace(u_a_mtx @ OBSERVABLE)
	return result

def get_Ham(n, EQCOUPLINGS, cp_type, **kwargs):
	if cp_type == 'dp':
		if EQCOUPLINGS:
			A = np.ones(n)
			A *= kwargs['alpha']
		else: #Only consider unequal case, which is the case of real systems
			A = np.random.uniform(0.5, 5, n)
			A /= 1000 #Scale to desired strength

		Delta = np.linspace(1.0,1.0+0.1*(n-1), num=n) #TLS frequencies
		Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency
		# compute Hilbert space basis
		basis = spin_basis_1d(L = n+1)
		# compute site-coupling lists
		z_term = [[-Delta[i]/2, i] for i in range(n+1)]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n)]

		#operator string lists
		static_d = [['z', z_term], 
					['-+', couple_term], ['+-', couple_term]]
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128, check_herm=False, check_symm=False)
	elif cp_type == 'Breuer_dp':
		A = np.ones(n)
		A *= kwargs['alpha']
		basis = spin_basis_1d(L = n+1)
		couple_term = [[A[i]/2, 0, i+1] for i in range(n)]
		static_d = [['-+', couple_term], ['+-', couple_term]]
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128, check_herm=False, check_symm=False)
	elif cp_type == 'iso':
		if EQCOUPLINGS:
			A = np.ones(n)
		else:
			A = np.random.uniform(1.0, 2.0, n) #According to Arenz
			# A = np.random.normal(1.0, 0.25, n)

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n+1)
		
		# compute site-coupling lists
		z_term = [[-0.5, 0]]
		couple_term = [[A[i], 0, i+1] for i in range(n)]
		x_term = [[1.0, 0]]

		#operator string lists
		static_d = [['z', z_term], ['zz', couple_term], 
					['xx', couple_term], ['yy', couple_term]]
		# static_c = [['x', x_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
	elif cp_type == 'Breuer_iso':
		A = np.ones(n)
		A *= kwargs['A']
		basis = spin_basis_1d(L = n+1)
		z_term = [[0.5, 0]]
		couple_term = [[A[i], 0, i+1] for i in range(n)]
		static_d = [['z', z_term], ['zz', couple_term], 
					['xx', couple_term], ['yy', couple_term]]
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)

	return H_d.toarray()

def v2rho(v3, vpm):
	'''
	Converts a given Bloch vector to density matrix
	'''
	return np.array([[(1+v3)/2, vpm[1]], [vpm[0], (1-v3)/2]])

# def main():
# 	n_max = 10
# 	EQCOUPLINGS = True
# 	t_f = 2
# 	T = np.linspace(0,t_f,200)
# 	OBSERVABLE = X
# 	cp_type = 'iso'
# 	b_init = 'zero'
# 	for n in range(n_max):
# 		print('n= ', n)
		
# 		H = get_Ham(n, EQCOUPLINGS, cp_type, )

# 		if b_init == 'zero':
# 			bath_init = np.zeros((2**n, 2**n), dtype='complex128')
# 			bath_init[0,0] = 1
# 		else:
# 			bath_init = np.identity(2**n, dtype='complex128') #unpolarized state of bath
# 			bath_init /= 2**n #Normalize
# 		print(np.trace(bath_init))

# 		sys_init = np.array([1.+0.0j, 0.0+1.0j])/np.sqrt(2)
# 		sys_init = np.outer(sys_init,sys_init.conj().T)
# 		print(sys_init)

# 		psi0 = np.kron(sys_init,bath_init)
# 		result_n =[]
# 		for t in T:
# 			result_n.append(evolveNmeasure(psi0, H, t, OBSERVABLE))
# 		plt.plot(T, result_n, label = n)

# 	plt.legend()
# 	plt.show()

# main()