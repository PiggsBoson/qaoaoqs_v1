from qaoaoqs.quantum_manager import *
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
sigma_y = 1.0j * np.array([[0.0, - 1.0], [1.0, 0.0]])

def Zrot(theta):
	#Returns z-rotation matrix with a given angle theta
	return np.array([[1, 0], [0, np.exp(1.0j*theta)]])

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

def setup(args, if_no_bath = False, couplings = None):
	'''Return the quantum manager corresponding the test case'''
	# test 2: Marin Bukov's paper
	if if_no_bath or (args.testcase == 'NoBath'):
		dyna_type = 'cs'
		fid_type = 'au'
		n=0
		A = np.ones(1)

		H0 = -0.5*sigma_z + 2.0*sigma_x
		H1 = -0.5*sigma_z - 2.0*sigma_x
		n_system = 1
		
	elif args.testcase == 'marin':
		psi0 = np.array([-1. / 2 - np.sqrt(5.) / 2, 1])
		psi0 = psi0 / la.norm(psi0)
		psi0_input = psi0.astype(complex)

		psi1 = np.array([1. / 2 + np.sqrt(5.) / 2, 1])
		psi1 = psi1 / la.norm(psi1)
		psi1_input = psi1.astype(complex)

		# sign is flip?
		H0 = (-sigma_z - (4.0) * sigma_x) / 2.
		H1 = (-sigma_z - (- 4.0) * sigma_x) / 2.
		n_system = 1

	#Central spin with state transfer.
	elif args.testcase == 'cs_st':
		dyna_type = 'cs'
		fid_type = 'st'
		n = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n)
		elif args.cs_coup == 'uneq':
			A = np.random.normal(1.0, 0.25, n)

		if args.impl == 'numpy':
			psi0 = np.kron(np.array([-1. / 2 - np.sqrt(5.) / 2, 1]), np.array([.5, .5])) #the bath qubit initial state, can be modified
			psi0 = psi0 / la.norm(psi0)
			psi0_input = psi0.astype(complex)

			psi1 = np.array([1. / 2 + np.sqrt(5.) / 2, 1])
			psi1 = psi1 / la.norm(psi1)
			psi1_input = psi1.astype(complex)

			#The coupling Hamiltonian
			HAB = A*(np.kron(sigma_x, sigma_x) + np.kron(sigma_y, sigma_y) + np.kron(sigma_z, sigma_z))
			HA = np.kron(-sigma_z, np.identity(2*n)) #The system Hamiltonian
			H00 = HA + HAB
			H0 = (H00 + (2.0) * np.kron(-sigma_x, np.identity(2*n))) / 2.
			H1 = (H00 - (2.0) * np.kron(-sigma_x, np.identity(2*n))) / 2.
		
		elif args.impl == 'quspin':

			# compute Hilbert space basis
			basis = spin_basis_1d(L = n+1)
			
			# compute site-coupling lists
			z_term = [[-0.5, 0]]
			couple_term = [[A[i], 0, i+1] for i in range(n)]
			x_term = [[1.0, 0]]

			#operator string lists
			static_d = [['z', z_term], ['zz', couple_term], 
						['xx', couple_term], ['yy', couple_term]]
			static_c = [['x', x_term]]

			#The drifting Hamiltonian
			H_d = hamiltonian(static_d, [], basis=basis, dtype=np.float64)
			#The control Hamiltonian
			H_c = hamiltonian(static_c, [], basis=basis, dtype=np.float64)

			#QAOA Hamiltonians
			H0 = H_d.toarray() + 2.0 * H_c.toarray()
			H1 = H_d.toarray() - 2.0 * H_c.toarray()

			#Generate states
			Hi = H_d.toarray() + H_c.toarray()
			Hf = H_d.toarray() - H_c.toarray()
			wi, vi = la.eig(Hi)
			wf, vf = la.eig(Hf)
			idxi = np.argmin(wi)
			psi_i = vi[:, idxi]
			psi0_input = psi_i.astype(complex)
			idxf = np.argmin(wf)
			psi_f = vf[:, idxf]
			psi1_input = psi_f.astype(complex)
		n_system = 1

	
	#Central spin with arbitrary unitary fidelity
	elif args.testcase == 'cs_au':
		dyna_type = 'cs'
		fid_type = 'au'
		n = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n)
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(1.0, 2.0, n) #According to Arenz
			# A = np.random.normal(1.0, 0.25, n)
		
		if args.impl == 'numpy':
			# import Wastes.CentralSpin as CentralSpin
			# H_d = CentralSpin.Hamiltonian(-0.5*sigma_z, A)
			# H_c = np.kron(sigma_x, np.identity(2**n))
			# #QAOA Hamiltonians
			# H0 = H_d + 2.0 * H_c
			# H1 = H_d - 2.0 * H_c
			pass

		elif args.impl == 'quspin':
			

			# compute Hilbert space basis
			basis = spin_basis_1d(L = n+1)
			
			# compute site-coupling lists
			z_term = [[-0.5, 0]]
			couple_term = [[A[i], 0, i+1] for i in range(n)]
			x_term = [[1.0, 0]]

			#operator string lists
			static_d = [['z', z_term], ['zz', couple_term], 
						['xx', couple_term], ['yy', couple_term]]
			static_c = [['x', x_term]]

			#The drifting Hamiltonian
			H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
			#The control Hamiltonian
			H_c = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)

			#QAOA Hamiltonians
			H0 = H_d.toarray() + 2.0 * H_c.toarray()
			H1 = H_d.toarray() - 2.0 * H_c.toarray()
		n_system = 1

	elif args.testcase == 'Heis_bbzz':
		'''
		Isotropic coupling with bath-bath couplings
		'''
		dyna_type = 'cs'
		fid_type = 'au'
		n = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n)
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(1.0, 2.0, n) #According to Arenz
			# A = np.random.normal(1.0, 0.25, n)
		
		# compute Hilbert space basis
		basis = spin_basis_1d(L = n+1)
		
		# compute site-coupling lists
		z_term = [[-0.5, 0]]
		couple_term = [[A[i], 0, i+1] for i in range(n)]
		x_term = [[1.0, 0]]

		g_bb = 0.001
		bb_term = [[g_bb, i+1, j+1]  for i in range(n) for j in range (i+1,n)]

		#operator string lists
		static_d = [['z', z_term], ['zz', couple_term], 
					['xx', couple_term], ['yy', couple_term],
					['zz', bb_term]]
		static_c = [['x', x_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonian
		H_c = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + 2.0 * H_c.toarray()
		H1 = H_d.toarray() - 2.0 * H_c.toarray()
		n_system = 1

	elif args.testcase == 'ns_cs':
		dyna_type = 'cs'
		fid_type = 'au'

		n = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n)
		elif args.cs_coup == 'uneq':
			A = np.random.normal(1.0, 0.25, n)

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n+1)
		
		# compute site-coupling lists
		z_term = [[-0.5, 0]]
		couple_term = [[A[i], 0, i+1] for i in range(n)]
		x_term = [[1.0, 0]]

		#operator string lists
		static_d = [['z', z_term], ['zz', couple_term], 
					['xx', couple_term], ['yy', couple_term]]
		static_c = [['x', x_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonian
		H_c = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + 2.0 * H_c.toarray()
		H1 = H_d.toarray() - 2.0 * H_c.toarray()
		n_system = 1

	elif args.testcase == 'dipole':
		dyna_type = 'cs'
		fid_type = 'au'
		n = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n)
		elif args.cs_coup == 'uneq':
			A = np.random.normal(1.0, 0.25, n)
		
		# compute Hilbert space basis
		basis = spin_basis_1d(L = n+1)
		
		# compute site-coupling lists
		z_term = [[-0.5, 0]]
		couple_term = [[A[i], 0, i+1] for i in range(n)]
		x_term = [[1.0, 0]]

		#operator string lists
		static_d = [['z', z_term], 
					['-+', couple_term], ['+-', couple_term]]
		static_c = [['x', x_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonian
		H_c = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + 2.0 * H_c.toarray()
		H1 = H_d.toarray() - 2.0 * H_c.toarray()
		n_system = 1
			
	elif args.testcase == 'XmonTLS':
		dyna_type = 'cs'
		fid_type = 'au'
		n = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n)
			A /= 200
		elif args.cs_coup == 'uneq': #Only consider unequal case, which is the case of real systems
			A = np.random.uniform(0.5, 5, n)
			A /= 1000 #Scale to desired strength
		
		Delta = np.linspace(1.0,1.0+0.1*(n-1), num=n) #TLS frequencies
		Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n+1)
		
		# compute site-coupling lists
		z_term = [[-Delta[i]/2, i] for i in range(n+1)]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n)]
		x_term = [[1.0, 0]]

		#operator string lists
		static_d = [['z', z_term], 
					['-+', couple_term], ['+-', couple_term]]
		static_c = [['x', x_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonian
		H_c = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + 2.0 * H_c.toarray()
		H1 = H_d.toarray() - 2.0 * H_c.toarray()
		n_system = 1

	elif args.testcase == 'TLS_bb':
		'''
		TLS-TLS coupling zz
		Towards understanding two-level-systems in amorphous solids - Insights from quantum circuits
		'''
		dyna_type = 'cs'
		fid_type = 'au'
		n = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n)
			A /= 200
		elif args.cs_coup == 'uneq': #Only consider unequal case, which is the case of real systems
			A = np.random.uniform(0.5, 5, n)
			A /= 1000 #Scale to desired strength
		
		Delta = np.linspace(1.0,1.0+0.1*(n-1), num=n) #TLS frequencies
		Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n+1)
		
		# compute site-coupling lists
		z_term = [[-Delta[i]/2, i] for i in range(n+1)]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n)]
		x_term = [[1.0, 0]]

		g_bb = 0.001
		bb_term = [[g_bb, i+1, j+1]  for i in range(n) for j in range (i+1,n)]
		#operator string lists
		static_d = [['z', z_term], 
					['-+', couple_term], ['+-', couple_term],
					['zz', bb_term]]
		static_c = [['x', x_term]]

		
		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonian
		H_c = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + 2.0 * H_c.toarray()
		H1 = H_d.toarray() - 2.0 * H_c.toarray()
		n_system = 1

	elif args.testcase =='TLSsec_bath':
		'''Exploiting Non-Markovianity for Quantum Control'''
		n_system = 1
		n = args.env_dim #number of bath qubits
		N = 2**(n+n_system)
		if args.cs_coup == 'eq':
			A = np.ones(n)
			A /= 200
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.5, 5, n)
			A /= 1000 #Scale to desired strength

		Delta = np.linspace(1.0,1.0+0.1*(n-1), num=n) #TLS frequencies
		Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency
		# compute Hilbert space basis
		basis = spin_basis_1d(L = n+1)
		# compute site-coupling lists
		z_term = [[-Delta[i]/2, i] for i in range(n+1)]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n)]
		x_term = [[1.0, 0]]
		#operator string lists
		static_d = [['z', z_term], 
					['-+', couple_term], ['+-', couple_term]]
		static_c = [['x', x_term]]
		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonian
		H_c = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)
		#QAOA Hamiltonians
		H0 = H_d.toarray() + 2.0 * H_c.toarray()
		H1 = H_d.toarray() - 2.0 * H_c.toarray()

		decoherence_type = args.deco_type
		gamma = [np.sqrt(1/args.T1_sys)] + [np.sqrt(1/args.T1_TLS)]*n 
		L = []
		for i in range(n+n_system):
			L_term = [[decoherence_type, [[gamma[i], i]] ]]
			H_temp = hamiltonian(L_term, [], basis=basis, dtype=np.complex128, check_herm=False)
			L.append(np.array(H_temp.toarray()))
		if args.impl =='qutip':
			H0 = qt.Qobj(H0)
			H1 = qt.Qobj(H1)
			L = [qt.Qobj(Li) for Li in L]

		dyna_type = 'lind_new'
		fid_type = 'GRK'

		psi1 =  target_uni(args.au_uni)
		psi1_input = psi1.astype(complex)
		if len(psi1_input) < 2**n_system:
			'''Append identity if dimension doesn't match'''
			Id = np.identity(2**n_system // len(psi1_input))
			psi1_input = np.kron(psi1_input, Id)
		psi0_input = None #Not used
		
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		quma = QuManager(psi0_input, psi1_input, H0, H1, dyna_type, fid_type, args,
						couplings = A, lind_L = L, n_s = n_system)

	elif args.testcase == 'Xmon_nb':
		dyna_type = 'cs'
		fid_type = 'au'
		n = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n)
			A /= 200
		elif args.cs_coup == 'uneq': #Only consider unequal case, which is the case of real systems
			A = np.random.uniform(0.5, 5, n)
			A /= 1000 #Scale to desired strength

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n+1)
		
		# compute site-coupling lists
		z_term = [[-0.5, 0]]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n)]
		x_term = [[1.0, 0]]

		#operator string lists
		static_d = [['z', z_term], 
					['-+', couple_term], ['+-', couple_term]]
		static_c = [['x', x_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonian
		H_c = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + 2.0 * H_c.toarray()
		H1 = H_d.toarray() - 2.0 * H_c.toarray()
		n_system = 1
	
	elif args.testcase == 'dipole_VarStr':
		dyna_type = 'cs'
		fid_type = 'au'
		n = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n)
		elif args.cs_coup == 'uneq':
			A = np.random.normal(1.0, 0.25, n)
		
		A /= 10**(args.cp_str) #sacle to desired strength, power of 10

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n+1)
		
		# compute site-coupling lists
		z_term = [[-0.5, 0]]
		couple_term = [[A[i], 0, i+1] for i in range(n)]
		x_term = [[1.0, 0]]

		#operator string lists
		static_d = [['z', z_term], 
					['-+', couple_term], ['+-', couple_term]]
		static_c = [['x', x_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonian
		H_c = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + 2.0 * H_c.toarray()
		H1 = H_d.toarray() - 2.0 * H_c.toarray()
		n_system = 1
	
	elif args.testcase == '2qbiSWAP':
		dyna_type = 'cs'
		fid_type = 'au'
		n1 = args.env_dim #number of bath qubits coupled to qubit 1
		n2 = args.env_dim2 #number of bath qubits coupled to qubit 2

		if args.cs_coup == 'eq':
			A = np.ones(n1+n2)
			A /= 200
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.5, 5, n1+n2)
			A /= 1000 #Scale to desired strength
		
		# E = np.linspace(1.0,1.0+0.1*(n-1), num=2) #TODO:qubit frequency
		g = 2.5/1000 #qubit-qubit coupling constant

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n1+n2+2)
		
		# compute site-coupling lists
		couple_term_1 = [[A[i]/2, 0, i+2] for i in range(n1)]
		couple_term_2 = [[A[i+n1]/2, 1, i+2+n1] for i in range(n2)]
		x_term = [[1.0, i] for i in range(2)]
		entangle_term = [[g/2, 0, 1]]

		#operator string lists
		static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
					['-+', couple_term_2], ['+-', couple_term_2]]
		control_a = [['x', x_term]]
		control_b = [['-+', entangle_term], ['+-', entangle_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonians
		H_ca = hamiltonian(control_a, [], basis=basis, dtype=np.complex128)
		H_cb = hamiltonian(control_b, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + H_ca.toarray()
		H1 = H_d.toarray() + H_cb.toarray()

		n=n1+n2
		n_system = 2

	elif args.testcase == '2qbCPHASE':
		dyna_type = 'cs'
		fid_type = 'au'
		n1 = args.env_dim #number of bath qubits coupled to qubit 1
		n2 = args.env_dim2 #number of bath qubits coupled to qubit 2
		
		if args.cs_coup == 'eq':
			A = np.ones(n1+n2)
			A /= 200
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.5, 5, n1+n2)
			A /= 1000 #Scale to desired strength
		
		# E = np.linspace(1.0,1.0+0.1*(n-1), num=2) #TODO:qubit frequency
		g = 2.5/1000 #qubit-qubit coupling constant		 

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n1+n2+2)
		
		# compute site-coupling lists
		couple_term_1 = [[A[i]/2, 0, i+2] for i in range(n1)]
		couple_term_2 = [[A[i+n1]/2, 1, i+2+n1] for i in range(n2)]
		x_term = [[1.0, i] for i in range(2)]
		entangle_term = [[- g/2, 0, 1]]
		z_term = [[g/2, i] for i in range(2)]

		#operator string lists
		static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
					['-+', couple_term_2], ['+-', couple_term_2]]
		control_a = [['x', x_term]]
		control_b = [['zz', entangle_term], ['z', z_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonians
		H_ca = hamiltonian(control_a, [], basis=basis, dtype=np.complex128)
		H_cb = hamiltonian(control_b, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + H_ca.toarray()
		H1 = H_d.toarray() + H_cb.toarray()
		n=n1+n2
		n_system = 2
	
	elif args.testcase == 'Lloyd_2qb':
		'''Proved to be universal:
		https://arxiv.org/abs/1812.11075v1
		On the universality of the quantum approximate optimization algorithm'''
		dyna_type = 'cs'
		fid_type = 'au'
		n1 = args.env_dim #number of bath qubits coupled to qubit 1
		n2 = args.env_dim2 #number of bath qubits coupled to qubit 2
		n_system = 2
		
		if args.cs_coup == 'eq':
			A = np.ones(n1+n2)
			A /= 200
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.5, 5, n1+n2)
			A /= 1000 #Scale to desired strength
		
		# E = np.linspace(1.0,1.0+0.1*(n-1), num=2) #TODO:qubit frequency
		g = 1.0 #qubit-qubit coupling constant

		 

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n1+n2+2)
		
		# compute site-coupling lists
		couple_term_1 = [[A[i]/2, 0, i+2] for i in range(n1)]
		couple_term_2 = [[A[i+n1]/2, 1, i+2+n1] for i in range(n2)]
		x_term = [[1.0, i] for i in range(2)]
		entangle_term = [[- g/2, 0, 1]]
		z_term = [[1.0, 0], [1.05, 1]]

		#operator string lists
		static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
					['-+', couple_term_2], ['+-', couple_term_2]]
		control_a = [['x', x_term]]
		control_b = [['zz', entangle_term], ['z', z_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonians
		H_ca = hamiltonian(control_a, [], basis=basis, dtype=np.complex128)
		H_cb = hamiltonian(control_b, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + H_ca.toarray()
		H1 = H_d.toarray() + H_cb.toarray()
		n=n1+n2

	elif args.testcase == 'Lloyd_var1':
		'''Proved to be universal:
		https://arxiv.org/abs/1812.11075v1
		On the universality of the quantum approximate optimization algorithm
		Change to a more practical scenario'''
		dyna_type = 'cs'
		fid_type = 'au'
		n1 = args.env_dim #number of bath qubits coupled to qubit 1
		n2 = args.env_dim2 #number of bath qubits coupled to qubit 2
		n_system = 2
		
		if args.cs_coup == 'eq':
			A = np.ones(n1+n2)
			A /= 200
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.5, 5, n1+n2)
			A /= 1000 #Scale to desired strength
		
		# E = np.linspace(1.0,1.0+0.1*(n-1), num=2) #TODO:qubit frequency
		g = 1.0 #qubit-qubit coupling constant

		 

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n1+n2+n_system)
		
		# compute site-coupling lists
		couple_term_1 = [[A[i]/2, 0, i+2] for i in range(n1)]
		couple_term_2 = [[A[i+n1]/2, 1, i+2+n1] for i in range(n2)]
		x_term = [[1.0, i] for i in range(2)]
		entangle_term = [[- g/2, 0, 1]]
		z_term = [[1.0, 0], [1.05, 1]]

		#operator string lists
		static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
					['-+', couple_term_2], ['+-', couple_term_2]]
		control = [['x', x_term]]
		static_sys = [['zz', entangle_term], ['z', z_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonians
		H_c = hamiltonian(control, [], basis=basis, dtype=np.complex128)
		H_s = hamiltonian(static_sys, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + H_s.toarray() + H_c.toarray()
		H1 = H_d.toarray() + H_s.toarray() - H_c.toarray()
		n=n1+n2

	elif args.testcase == 'Lloyd_3qb':
		'''Proved to be universal:
		[1] https://arxiv.org/abs/1812.11075v1
		[2] On the universality of the quantum approximate optimization algorithm
		According to [2], need odd number of qubits for symmetry breaking. So add one ancilla'''
		dyna_type = 'cs'
		fid_type = 'au'
		n1 = args.env_dim #number of bath qubits coupled to qubit 1
		n2 = args.env_dim2 #number of bath qubits coupled to qubit 2
		n_system = 3

		if args.cs_coup == 'eq':
			A = np.ones(n1+n2)
			A /= 200
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.5, 5, n1+n2)
			A /= 1000 #Scale to desired strength
		
		# E = np.linspace(1.0,1.0+0.1*(n-1), num=2) #TODO:qubit frequency
		g = [1.1, 1.15] #qubit-qubit coupling constants

		 

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n1+n2+n_system)

		
		# compute site-coupling lists
		couple_term_1 = [[A[i]/2, 0, i+n_system] for i in range(n1)]
		couple_term_2 = [[A[i+n1]/2, 1, i+n_system+n1] for i in range(n2)]
		x_term = [[1.0, i] for i in range(n_system)]
		entangle_term = [[- g[0]/2, 0, 1], [- g[1]/2, 1, 2]]
		z_term = [[1.0, 0], [1.05, 1], [1.0,2]]

		#operator string lists
		static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
					['-+', couple_term_2], ['+-', couple_term_2]]
		control_a = [['x', x_term]]
		control_b = [['zz', entangle_term], ['z', z_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonians
		H_ca = hamiltonian(control_a, [], basis=basis, dtype=np.complex128)
		H_cb = hamiltonian(control_b, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + H_ca.toarray()
		H1 = H_d.toarray() + H_cb.toarray()
		n=n1+n2

	elif args.testcase == 'XXnYY':
		dyna_type = 'cs'
		fid_type = 'au'
		n1 = args.env_dim #number of bath qubits coupled to qubit 1
		n2 = args.env_dim2 #number of bath qubits coupled to qubit 2
		
		if args.cs_coup == 'eq':
			A = np.ones(n1+n2)
			A /= 200
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.5, 5, n1+n2)
			A /= 1000 #Scale to desired strength

		# E = np.linspace(1.0,1.0+0.1*(n-1), num=2) #TODO:qubit frequency
		g = 2.5/1000 #qubit-qubit coupling constant 

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n1+n2+2)
		
		# compute site-coupling lists
		couple_term_1 = [[A[i]/2, 0, i+2] for i in range(n1)]
		couple_term_2 = [[A[i+n1]/2, 1, i+2+n1] for i in range(n2)]
		entangle_term = [[g/2, 0, 1]]

		#operator string lists
		if hasattr(args,'cs_coup_type'): #accomondate old versions
			if args.cs_coup_type == 'Heis':
				static_d = [['xx', couple_term_1], ['yy', couple_term_1], ['zz', couple_term_1],
							['xx', couple_term_2], ['yy', couple_term_2], ['zz', couple_term_2]]
			else:
				static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
						['-+', couple_term_2], ['+-', couple_term_2]]
		else:
			#dipole-dipole is a default
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
		n=n1+n2
		n_system = 2
	
	elif args.testcase == 'XXYY_X':
		'''Adding a single qubit X to counteract decoherence.'''
		dyna_type = 'cs'
		fid_type = 'au'
		n1 = args.env_dim #number of bath qubits coupled to qubit 1
		n2 = args.env_dim2 #number of bath qubits coupled to qubit 2
		
		if args.cs_coup == 'eq':
			A = np.ones(n1+n2)
			A /= 200
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.5, 5, n1+n2)
			A /= 1000 #Scale to desired strength

		# E = np.linspace(1.0,1.0+0.1*(n-1), num=2) #TODO:qubit frequency
		g = 2.5/1000 #qubit-qubit coupling constant

		 

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n1+n2+2)
		
		# compute site-coupling lists
		couple_term_1 = [[A[i]/2, 0, i+2] for i in range(n1)]
		couple_term_2 = [[A[i+n1]/2, 1, i+2+n1] for i in range(n2)]
		entangle_term = [[g/2, 0, 1]]
		x_term = [[-0.5, 0]]

		#operator string lists
		if hasattr(args,'cs_coup_type'): #accomondate old versions
			if args.cs_coup_type == 'Heis':
				static_d = [['xx', couple_term_1], ['yy', couple_term_1], ['zz', couple_term_1],
							['xx', couple_term_2], ['yy', couple_term_2], ['zz', couple_term_2]]
			else:
				static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
						['-+', couple_term_2], ['+-', couple_term_2]]
		else:
			#dipole-dipole is a default
			static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
						['-+', couple_term_2], ['+-', couple_term_2]]
		control_a = [['xx', entangle_term], ['yy', entangle_term]]
		control_b = [['x', x_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonians
		H_ca = hamiltonian(control_a, [], basis=basis, dtype=np.complex128)
		H_cb = hamiltonian(control_b, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + H_ca.toarray()
		H1 = H_d.toarray() + H_cb.toarray()
		n=n1+n2
		n_system = 2

	elif args.testcase == 'XXpm':
		'''2-qubit XX. 
		From 'Environment-invariant measure of distance between evolutions of an open quantum system' 
		Seems to be wrong. Not contiuning.
		'''
		dyna_type = 'cs'
		fid_type = 'au'
		n1 = args.env_dim #number of bath qubits coupled to qubit 1
		n2 = args.env_dim2 #number of bath qubits coupled to qubit 2
		
		if args.cs_coup == 'eq':
			A = np.ones(n1+n2)
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.1, 1, n1+n2)

		# E = [1.0, 1.05] #TODO:qubit frequency
		g = 1.0 #qubit-qubit coupling constant

		 

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n1+n2+2)
		
		# compute site-coupling lists
		sys_term = [[1.0/2, 0], [1.05/2, 1]]
		couple_term_1 = [[A[i]/2, 0, i+2] for i in range(n1)]
		couple_term_2 = [[A[i+n1]/2, 1, i+2+n1] for i in range(n2)]
		entangle_term = [[g/2, 0, 1]]

		#operator string lists
		if hasattr(args,'cs_coup_type'): #accomondate old versions
			if args.cs_coup_type == 'Heis':
				static_d = [['xx', couple_term_1], ['yy', couple_term_1], ['zz', couple_term_1],
							['xx', couple_term_2], ['yy', couple_term_2], ['zz', couple_term_2]]
			else:
				static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
						['-+', couple_term_2], ['+-', couple_term_2]]
		else:
			#dipole-dipole is a default
			static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
						['-+', couple_term_2], ['+-', couple_term_2]]
		static_d.append(['z', sys_term])
		control_a = [['xx', entangle_term]]

		
		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonians
		H_c = hamiltonian(control_a, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + H_c.toarray()
		H1 = H_d.toarray() - H_c.toarray()
		n=n1+n2
		n_system = 2

	elif args.testcase == 'result_analysis':
		'''Use this with caution!!!'''
		dyna_type = 'cs'
		fid_type = 'au'
		n1 = args.env_dim #number of bath qubits coupled to qubit 1
		n2 = args.env_dim2 #number of bath qubits coupled to qubit 2
		n_system = 2
		
		A = np.array(couplings, ndmin=1)
		A /= args.cs_coup_scale #Make sure it's not scaled twice
		g = 2.5/1000 #qubit-qubit coupling constant

		 

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n1+n2+n_system)
		#TODO: this now only works for XXnYY
		# compute site-coupling lists
		couple_term_1 = [[A[i]/2, 0, i+2] for i in range(n1)]
		couple_term_2 = [[A[i+n1]/2, 1, i+2+n1] for i in range(n2)]
		entangle_term = [[g/2, 0, 1]]

		#operator string lists
		if hasattr(args,'cs_coup_type'): #accomondate old versions
			if args.cs_coup_type == 'Heis':
				static_d = [['xx', couple_term_1], ['yy', couple_term_1], ['zz', couple_term_1],
							['xx', couple_term_2], ['yy', couple_term_2], ['zz', couple_term_2]]
			else:
				static_d = [['-+', couple_term_1], ['+-', couple_term_1], 
						['-+', couple_term_2], ['+-', couple_term_2]]
		else:
			#dipole-dipole is a default
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
		n=n1+n2

	#Set up target unitary
	if args.testcase != ('lind' or 'ns_lind' or 'TLSsec_bath'):
		psi1 =  target_uni(args.au_uni)
		psi1_input = psi1.astype(complex)
		if len(psi1_input) < 2**n_system:
			'''Append identity if dimension doesn't match'''
			Id = np.identity(2**n_system // len(psi1_input))
			psi1_input = np.kron(psi1_input, Id)
		psi0_input = np.identity(2**(n+n_system)) #Start from identity
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		quma = QuManager(psi0_input, psi1_input, H0, H1, dyna_type, fid_type, args, couplings = A)
	
	return quma