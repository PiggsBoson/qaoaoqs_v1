#Check if it works for 2 qubits. Also test fidelity TODO
import numpy as np
import scipy.linalg as la
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/william/OneDrive - HKUST Connect/Codes/QAOA/PGQAOA_OpenQuantumSystem/')
from quantum_manager import *
import matplotlib.pyplot as plt

def plotmtx(M):
    plt.matshow(np.real(M))
    plt.colorbar()
    plt.matshow(np.imag(M))
    plt.colorbar()
    plt.show()

def ptrace(rho, remain, dim_a, dim_b):
    #The following is partial trace according to Peijun Zhu https://www.peijun.me/reduced-density-matrix-and-partial-trace.html
    rho_tensor = rho.reshape([dim_a, dim_b, dim_a, dim_b])
    if remain == 'a':
        return np.trace(rho_tensor, axis1=1, axis2=3)
    elif remain == 'b':
        return np.trace(rho_tensor, axis1=0, axis2=2)

def target_uni(uni_name):
	#Returns the target unitary given name
	if uni_name == 'Had':
		return 1/np.sqrt(2)*np.array([[1.0, 1.0], [1.0, -1.0]])
	elif uni_name == 'ST':
		return -1. * sigma_z
	# elif uni_name == 'Zrot':
	#     theta = np.pi * args.target_angle
	#     psi1 = np.array([[np.exp(-1.0j*theta/2), 0], [0, np.exp(1.0j*theta/2)]])
	elif uni_name == 'T': #T gate
		return Zrot(np.pi/4)
	elif uni_name == 'S': #S gate
		return Zrot(np.pi/2)
	elif uni_name == 'CNOT': #CNOT gate
		return np.array([[1, 0, 0, 0], 
						[0, 1, 0, 0],
						[0, 0, 0, 1],
						[0, 0, 1, 0]])
	elif uni_name == 'iSWAP': #iSWAP gate
		return np.array([[1, 0, 0, 0], 
						[0, 0, -1.0j, 0],
						[0, -1.0j, 0, 0],
						[0, 0, 0, 1]])
	else:
		#Random SU(2) using Cayley-Klein parameterization
		x = np.random.rand(4)
		x = x / la.norm(x) #Normalize
		return x[0] * np.identity(2) + 1.0j * (x[1] * sigma_x + x[2] * sigma_y + x[3] * sigma_z) #The target unitary W on the system.

def fidelity(u, target, N_s, N_b):
    target = np.kron(target, np.identity(N_b))
    Q = np.matmul(target.conjugate().transpose(), u )
    Q_tensor = Q.reshape([N_s, N_b, N_s, N_b])
    Q_b = np.trace(Q_tensor, axis1=0, axis2=2) #partial trace over sys
    return np.absolute(np.trace(la.sqrtm(Q_b.conjugate().transpose() @ Q_b)) / (N_s*N_b)) ** 2

#Testcase 1
# target1 = target_uni('CNOT')
# u1 = np.kron(target1, target_uni('random'))
# # print(u1)
# print(fidelity(u1,target1,4, 2))

#Testcase 2
def testcase2(p = 20):
	dyna_type = 'cs'
	fid_type = 'au'
	n1 = 0 #number of bath qubits coupled to qubit 1
	n2 = 0 #number of bath qubits coupled to qubit 2

	A = np.random.uniform(0.5, 5, n1+n2)
	A /= 1000 #Scale to desired strength

	# E = np.linspace(1.0,1.0+0.1*(n-1), num=2) #TODO:qubit frequency
	g = 2.5/1000 #qubit-qubit coupling constant

	from quspin.basis import spin_basis_1d
	from quspin.operators import hamiltonian

	# compute Hilbert space basis
	basis = spin_basis_1d(L = n1+n2+2)

	# compute site-coupling lists
	couple_term_1 = [[A[i]/2, 0, i+2] for i in range(n1)]
	couple_term_2 = [[A[i+n1]/2, 1, i+2+n1] for i in range(n2)]
	entangle_term = [[g/2, 0, 1]]

	#operator string lists
	static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
				['-+', couple_term_2], ['+-', couple_term_2]]
	control_a = [['xx', entangle_term]]
	control_b = [['yy', entangle_term]]

	#The drifting Hamiltonian
	H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
	#The control Hamiltonians
	H_ca = hamiltonian(control_a, [], basis=basis, dtype=np.complex128)
	H_cb = hamiltonian(control_b, [], basis=basis, dtype=np.complex128)

	#QAOA Hamiltonians
	H0 = H_d.toarray() + H_ca.toarray()
	H1 = H_d.toarray() + H_cb.toarray()
	# print(H0)
	n=n1+n2
	n_system = 2

	psi1 =  target_uni('iSWAP')
	psi1_input = psi1.astype(complex)
	psi0_input = np.identity(2**(n+n_system)) 

	quma = QuManager(psi0_input, psi1_input, H0, H1, dyna_type, fid_type, couplings = A,
					testcase= 'XXnYY',env_dim=n1, env_dim2=n2)

	T = 1256.6370/2
	protocol_a = np.random.uniform(size = p)
	protocol_a /= np.sum(protocol_a)
	protocol_a *= T
	print(np.sum(protocol_a))
	protocol_b = np.random.uniform(size = p)
	protocol_b /= np.sum(protocol_b)
	protocol_b *= T
	print(np.sum(protocol_b))
	protocol = [item for pair in zip(protocol_a, protocol_b + [0]) 
							for item in pair]
	print(quma.get_reward(protocol))

def testcase3(p = 20):
	dyna_type = 'cs'
	fid_type = 'au'
	n1 = 2 #number of bath qubits coupled to qubit 1
	n2 = 2 #number of bath qubits coupled to qubit 2

	A = np.random.uniform(0.5, 5, n1+n2)
	A /= 1000 #Scale to desired strength

	# E = np.linspace(1.0,1.0+0.1*(n-1), num=2) #TODO:qubit frequency
	g = 2.5/1000 #qubit-qubit coupling constant

	from quspin.basis import spin_basis_1d
	from quspin.operators import hamiltonian

	# compute Hilbert space basis
	basis = spin_basis_1d(L = n1+n2+2)

	# compute site-coupling lists
	couple_term_1 = [[A[i]/2, 0, i+2] for i in range(n1)]
	couple_term_2 = [[A[i+n1]/2, 1, i+2+n1] for i in range(n2)]
	entangle_term = [[ g/2, 0, 1]]

	#operator string lists
	static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
				['-+', couple_term_2], ['+-', couple_term_2]]
	control_a = [['xx', entangle_term],['yy', entangle_term]]

	#The drifting Hamiltonian
	H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
	#The control Hamiltonians
	H_ca = hamiltonian(control_a, [], basis=basis, dtype=np.complex128)

	#QAOA Hamiltonians
	H0 = H_d.toarray() + H_ca.toarray()
	n=n1+n2
	n_system = 2

	psi1 =  target_uni('iSWAP')
	# psi1_input = psi1.astype(complex)
	psi0_input = np.identity(2**(n+n_system)) 

	quma = QuManager(psi0_input, psi1, H0, H0, dyna_type, fid_type, couplings = A,
					testcase= 'XXnYY',env_dim=n1, env_dim2=n2)

	T = np.pi / (2*g)
	dt = T/(2*p)
	protocol = np.ones(2*p) * dt
	print(quma.get_reward(protocol))
p=100
testcase2()
testcase3()