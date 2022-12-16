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

def setup(args, if_no_bath = False, couplings = None, alt_testcase = None, interaction_pic = False):
	'''Return the quantum manager corresponding the test case
	
	'''
	if couplings:
		A = np.array(couplings)
	if alt_testcase:
		test_case = alt_testcase
	else: 
		test_case = args.testcase
	print(test_case)
	# test 2: Marin Bukov's paper
	if if_no_bath or (test_case == 'NoBath'):
		dyna_type = 'cs'
		fid_type = 'au'
		n_b=0
		A = np.ones(1)

		H0 = -0.5*sigma_z + 2.0*sigma_x
		H1 = -0.5*sigma_z - 2.0*sigma_x
		n_system = 1
		
	elif test_case == 'marin':
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
	elif test_case == 'cs_st':
		dyna_type = 'cs'
		fid_type = 'st'
		n_b = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n_b)
		elif args.cs_coup == 'uneq':
			A = np.random.normal(1.0, 0.25, n_b)

		if args.impl == 'numpy':
			psi0 = np.kron(np.array([-1. / 2 - np.sqrt(5.) / 2, 1]), np.array([.5, .5])) #the bath qubit initial state, can be modified
			psi0 = psi0 / la.norm(psi0)
			psi0_input = psi0.astype(complex)

			psi1 = np.array([1. / 2 + np.sqrt(5.) / 2, 1])
			psi1 = psi1 / la.norm(psi1)
			psi1_input = psi1.astype(complex)

			#The coupling Hamiltonian
			HAB = A*(np.kron(sigma_x, sigma_x) + np.kron(sigma_y, sigma_y) + np.kron(sigma_z, sigma_z))
			HA = np.kron(-sigma_z, np.identity(2*n_b)) #The system Hamiltonian
			H00 = HA + HAB
			H0 = (H00 + (2.0) * np.kron(-sigma_x, np.identity(2*n_b))) / 2.
			H1 = (H00 - (2.0) * np.kron(-sigma_x, np.identity(2*n_b))) / 2.
		
		elif args.impl == 'quspin':

			# compute Hilbert space basis
			basis = spin_basis_1d(L = n_b+1)
			
			# compute site-coupling lists
			z_term = [[-0.5, 0]]
			couple_term = [[A[i], 0, i+1] for i in range(n_b)]
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
	elif test_case == 'cs_au':
		dyna_type = 'cs'
		fid_type = 'au'
		n_b = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n_b)
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(1.0, 2.0, n_b) #According to Arenz
			# A = np.random.normal(1.0, 0.25, n_b)
		
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		
		# compute site-coupling lists
		z_term = [[-0.5, 0]]
		couple_term = [[A[i], 0, i+1] for i in range(n_b)]
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
		if interaction_pic:
			static_Sd = [['z', z_term]]
			static_int = [['zz', couple_term], 
					['xx', couple_term], ['yy', couple_term]]
			H_Sd = hamiltonian(static_Sd, [], basis=basis, dtype=np.complex128)

			H_B = None
			H_int = hamiltonian(static_int, [], basis=basis, dtype=np.complex128).toarray()
			H_S0 =  H_Sd.toarray() + 2.0 * H_c.toarray()
			H_S1 = H_Sd.toarray() - 2.0 * H_c.toarray()

	elif test_case == 'Heis_lab':
		'''
		Heisenberg coupling in the lab frame (with bath spin energy)
		'''
		dyna_type = 'cs'
		fid_type = 'au'
		n_b = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n_b)
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(1.0, 2.0, n_b) #According to Arenz
			# A = np.random.normal(1.0, 0.25, n_b)
		
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)

		Delta = np.linspace(1.0,1.0+0.1*(n_b-1), num=n_b) #TLS frequencies
		Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency

		# compute site-coupling lists
		z_term = [[-Delta[i]/2, i] for i in range(n_b+1)]
		couple_term = [[A[i], 0, i+1] for i in range(n_b)]
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

	elif test_case == 'iso_4Ham':
		'''
		Introducing y terms trying to improve the fidelity
		'''
		dyna_type = 'cs'
		fid_type = 'au'
		n_b = args.env_dim #number of bath qubits
		n_system = 1

		if args.cs_coup == 'eq':
			A = np.ones(n_b)
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(1.0, 2.0, n_b) #According to Arenz
			# A = np.random.normal(1.0, 0.25, n_b)
		
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)

		Delta = np.linspace(1.0,1.0+0.1*(n_b-1), num=n_b) #TLS frequencies
		Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency

		# compute site-coupling lists
		z_term = [[-Delta[0]/2, i]]
		if args.bath_Z_terms:
			z_term += [[-Delta[i]/2, i] for i in range(1,n_b+n_system)]
		couple_term = [[A[i], 0, i+1] for i in range(n_b)]
		x_term = [[1.0, 0]]
		y_term = [[1.0, 0]]

		#operator string lists
		static_d = [['z', z_term], ['zz', couple_term], 
					['xx', couple_term], ['yy', couple_term]]
		static_cX = [['x', x_term]]
		static_cY = [['y', y_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonians
		H_cX = hamiltonian(static_cX, [], basis=basis, dtype=np.complex128)
		H_cY = hamiltonian(static_cY, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + 2.0 * H_cX.toarray() + 1.5 * H_cY.toarray()
		H1 = H_d.toarray() + 2.0 * H_cX.toarray() - 1.5 * H_cY.toarray()
		H2 = H_d.toarray() - 2.0 * H_cX.toarray() + 1.5 * H_cY.toarray()
		H3 = H_d.toarray() - 2.0 * H_cX.toarray() - 1.5 * H_cY.toarray()


	elif test_case == 'Heis_bbzz':
		'''
		Isotropic coupling with bath-bath couplings
		'''
		dyna_type = 'cs'
		fid_type = 'au'
		n_b = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n_b)
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(1.0, 2.0, n_b) #According to Arenz
			# A = np.random.normal(1.0, 0.25, n_b)
		
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		
		# compute site-coupling lists
		z_term = [[-0.5, 0]]
		couple_term = [[A[i], 0, i+1] for i in range(n_b)]
		x_term = [[1.0, 0]]

		g_bb = 0.001
		bb_term = [[g_bb, i+1, j+1]  for i in range(n_b) for j in range (i+1,n_b)]

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

	elif test_case == 'ns_cs':
		dyna_type = 'cs'
		fid_type = 'au'

		n_b = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n_b)
		elif args.cs_coup == 'uneq':
			A = np.random.normal(1.0, 0.25, n_b)
		
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		
		# compute site-coupling lists
		z_term = [[-0.5, 0]]
		couple_term = [[A[i], 0, i+1] for i in range(n_b)]
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

	elif test_case == 'dipole':
		dyna_type = 'cs'
		fid_type = 'au'
		n_b = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n_b)
		elif args.cs_coup == 'uneq':
			A = np.random.normal(1.0, 0.25, n_b)
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		
		# compute site-coupling lists
		z_term = [[-0.5, 0]]
		couple_term = [[A[i], 0, i+1] for i in range(n_b)]
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
			
	elif test_case == 'XmonTLS':
		dyna_type = 'cs'
		fid_type = 'au'
		n_b = args.env_dim #number of bath qubits
		if not couplings:
			if args.cs_coup == 'eq':
				A = np.ones(n_b)
				A /= 200
			elif args.cs_coup == 'uneq': #Only consider unequal case, which is the case of real systems
				A = np.random.uniform(0.5, 5, n_b)
				A /= 1000 #Scale to desired strength
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		Delta = np.linspace(1.0,1.0+0.1*(n_b-1), num=n_b) #TLS frequencies
		Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		
		# compute site-coupling lists
		z_term = [[-Delta[i]/2, i] for i in range(n_b+1)]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n_b)]
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
		if interaction_pic:
			static_Sd = [['z', z_term[:1]]]
			static_Bd = [['z', z_term[1:]]]
			static_int = [['-+', couple_term], ['+-', couple_term]]
			H_Sd = hamiltonian(static_Sd, [], basis=basis, dtype=np.complex128)

			H_B = hamiltonian(static_Bd, [], basis=basis, dtype=np.complex128).toarray()
			H_int = hamiltonian(static_int, [], basis=basis, dtype=np.complex128).toarray()
			H_S0 =  H_Sd.toarray() + 2.0 * H_c.toarray()
			H_S1 = H_Sd.toarray() - 2.0 * H_c.toarray()

	elif test_case == 'dp_4Ham':
		'''
		Introducing y terms trying to improve the fidelity
		'''
		dyna_type = 'cs'
		fid_type = 'au'
		n_b = args.env_dim #number of bath qubits
		n_system = 1
		if not couplings:
			if args.cs_coup == 'eq':
				A = np.ones(n_b)
				A /= 200
			elif args.cs_coup == 'uneq': #Only consider unequal case, which is the case of real systems
				A = np.random.uniform(0.5, 5, n_b)
				A /= 1000 #Scale to desired strength
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0

		Delta = np.linspace(1.0,1.0+0.1*(n_b-1), num=n_b) #TLS frequencies
		Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		
		# compute site-coupling lists
		z_term = [[-Delta[0]/2, i]]
		if args.bath_Z_terms:
			z_term += [[-Delta[i]/2, i] for i in range(1,n_b+n_system)]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n_b)]
		x_term = [[1.0, 0]]
		y_term = [[1.0, 0]]

		#operator string lists
		static_d = [['z', z_term], 
					['-+', couple_term], ['+-', couple_term]]
		static_cX = [['x', x_term]]
		static_cY = [['y', y_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonians
		H_cX = hamiltonian(static_cX, [], basis=basis, dtype=np.complex128)
		H_cY = hamiltonian(static_cY, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + 2.0 * H_cX.toarray() + 1.5 * H_cY.toarray()
		H1 = H_d.toarray() + 2.0 * H_cX.toarray() - 1.5 * H_cY.toarray()
		H2 = H_d.toarray() - 2.0 * H_cX.toarray() + 1.5 * H_cY.toarray()
		H3 = H_d.toarray() - 2.0 * H_cX.toarray() - 1.5 * H_cY.toarray()
		
	elif test_case == 'TLS_bb':
		'''
		TLS-TLS coupling zz
		Towards understanding two-level-systems in amorphous solids - Insights from quantum circuits
		'''
		dyna_type = 'cs'
		fid_type = 'au'
		n_b = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n_b)
			A /= 200
		elif args.cs_coup == 'uneq': #Only consider unequal case, which is the case of real systems
			A = np.random.uniform(0.5, 5, n_b)
			A /= 1000 #Scale to desired strength
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		Delta = np.linspace(1.0,1.0+0.1*(n_b-1), num=n_b) #TLS frequencies
		Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		
		# compute site-coupling lists
		z_term = [[-Delta[i]/2, i] for i in range(n_b+1)]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n_b)]
		x_term = [[1.0, 0]]

		g_bb = 0.001
		bb_term = [[g_bb, i+1, j+1]  for i in range(n_b) for j in range (i+1,n_b)]
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

	elif test_case =='TLSsec_bath':
		'''Exploiting Non-Markovianity for Quantum Control'''
		n_system = 1
		n_b = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n_b)
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.1, 1, n_b)
		A *= 0.04 #40MHz
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		Delta = np.linspace(1.0,1.0+0.1*(n_b-1), num=n_b) #TLS frequencies
		Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency
		Delta *= 8.0 #8GHz

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		# compute site-coupling lists
		z_term = [[-Delta[i]/2, i] for i in range(n_b+1)]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n_b)]
		x_term = [[8.0, 0]]
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
		gamma = [np.sqrt(1/args.T1_sys)] + [np.sqrt(1/args.T1_TLS)]*n_b 
		L = []
		for i in range(n_b+n_system):
			L_term = [[decoherence_type, [[gamma[i], i]] ]]
			H_temp = hamiltonian(L_term, [], basis=basis, dtype=np.complex128, check_herm=False)
			L.append(np.array(H_temp.toarray()))
		if args.impl =='qutip':
			H0 = qt.Qobj(H0)
			H1 = qt.Qobj(H1)
			L = [qt.Qobj(Li) for Li in L]

		dyna_type = 'lind_new'
		fid_type = 'GRK'

	elif test_case =='TLSsec_bath_lowStr':
		'''Exploiting Non-Markovianity for Quantum Control
		lowered strength, same as other test cases'''
		n_system = 1
		n_b = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n_b)
			A /= 200
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.5, 5, n_b)
			A /= 1000 #Scale to desired strength
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		Delta = np.linspace(1.0,1.0+0.1*(n_b-1), num=n_b) #TLS frequencies
		Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		# compute site-coupling lists
		z_term = [[-Delta[i]/2, i] for i in range(n_b+1)]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n_b)]
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
		gamma = [np.sqrt(1/args.T1_sys)] + [np.sqrt(1/args.T1_TLS)]*n_b
		gamma = [gamma_i/np.sqrt(8) for gamma_i in gamma] #Strength Scaling
		L = []
		for i in range(n_b+n_system):
			L_term = [[decoherence_type, [[gamma[i], i]] ]]
			H_temp = hamiltonian(L_term, [], basis=basis, dtype=np.complex128, check_herm=False)
			L.append(np.array(H_temp.toarray()))
		if args.impl =='qutip':
			H0 = qt.Qobj(H0)
			H1 = qt.Qobj(H1)
			L = [qt.Qobj(Li) for Li in L]

		dyna_type = 'lind_new'
		fid_type = 'GRK'
	
	elif test_case =='Koch_1qb':
		'''Exploiting Non-Markovianity for Quantum Control
		same ansatz, parameter strength is ours'''
		n_system = 1
		n_b = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n_b)
			A /= 200
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.5, 5, n_b)
			A /= 1000 #Scale to desired strength
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		Delta = np.linspace(1.0,1.0+0.1*(n_b-1), num=n_b) #TLS frequencies
		Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		# compute site-coupling lists
		z_term = [[-Delta[i]/2, i] for i in range(n_b+1)]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n_b)]
		ctr_term = [[1.0, 0]]
		#operator string lists
		static_d = [['z', z_term], 
					['-+', couple_term], ['+-', couple_term]]
		static_c = [['z', ctr_term]]
		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonian
		H_c = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)
		#QAOA Hamiltonians
		H0 = H_d.toarray() + 2.0 * H_c.toarray()
		H1 = H_d.toarray() - 2.0 * H_c.toarray()

		decoherence_type = args.deco_type
		gamma = [np.sqrt(1/args.T1_sys)] + [np.sqrt(1/args.T1_TLS)]*n_b
		gamma = [gamma_i/np.sqrt(8) for gamma_i in gamma] 
		L = []
		for i in range(n_b+n_system):
			L_term = [[decoherence_type, [[gamma[i], i]] ]]
			H_temp = hamiltonian(L_term, [], basis=basis, dtype=np.complex128, check_herm=False)
			L.append(np.array(H_temp.toarray()))
		if args.impl =='qutip':
			H0 = qt.Qobj(H0)
			H1 = qt.Qobj(H1)
			L = [qt.Qobj(Li) for Li in L]

		dyna_type = 'lind_new'
		fid_type = 'GRK'

	elif test_case == 'Koch_paper_1qb_noLind':
		'''Exploiting Non-Markovianity for Quantum Control
		exact match of their parameters but without Lindblad'''
		dyna_type = 'cs'
		fid_type = 'au'
		n_b = args.env_dim #number of bath qubits
		if not couplings:
			if args.cs_coup == 'eq':
				A = np.ones(n_b)
				A /= 200
			elif args.cs_coup == 'uneq': #Only consider unequal case, which is the case of real systems
				A = np.random.uniform(0.5, 5, n_b)
				A /= 1000 #Scale to desired strength
		A *= 0.06 #60MHz
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		Delta = [5.0, 5.0-0.55, 5.0-0.55-0.45] #Frequency list, 5GHz

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		
		# compute site-coupling lists
		z_term = [[-Delta[i]/2, i] for i in range(n_b+1)]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n_b)]
		ctr_term = [[5.0, 0]]

		#operator string lists
		static_d = [['z', z_term], 
					['-+', couple_term], ['+-', couple_term]]
		static_c = [['z',ctr_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonian
		H_c = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + 2*H_c.toarray()
		H1 = H_d.toarray() - 2*H_c.toarray()
		n_system = 1

	elif test_case =='Koch_paper_1qb':
		'''Exploiting Non-Markovianity for Quantum Control
		exact match of their parameters'''
		n_system = 1
		n_b = args.env_dim #number of bath qubits
		if not couplings:
			if args.cs_coup == 'eq':
				A = np.ones(n_b)
				A /= 200
			elif args.cs_coup == 'uneq': #Only consider unequal case, which is the case of real systems
				A = np.random.uniform(0.5, 5, n_b)
				A /= 1000 #Scale to desired strength
		A *= 0.06 #60MHz
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		Delta = [5.0, 5.0-0.55, 5.0-0.55-0.45] #Frequency list, 5GHz

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		
		# compute site-coupling lists
		z_term = [[-Delta[i]/2, i] for i in range(n_b+1)]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n_b)]
		ctr_term = [[5.0, 0]]

		#operator string lists
		static_d = [['z', z_term], 
					['-+', couple_term], ['+-', couple_term]]
		static_c = [['z',ctr_term]]

		#The drifting Hamiltonian
		H_d = hamiltonian(static_d, [], basis=basis, dtype=np.complex128)
		#The control Hamiltonian
		H_c = hamiltonian(static_c, [], basis=basis, dtype=np.complex128)

		#QAOA Hamiltonians
		H0 = H_d.toarray() + 2*H_c.toarray()
		H1 = H_d.toarray() - 2*H_c.toarray()

		decoherence_type = args.deco_type
		gamma = [np.sqrt(1/args.T1_sys)] + [np.sqrt(1/args.T1_TLS)]*n_b
		L = []
		for i in range(n_b+n_system):
			L_term = [[decoherence_type, [[gamma[i], i]] ]]
			H_temp = hamiltonian(L_term, [], basis=basis, dtype=np.complex128, check_herm=False)
			L.append(np.array(H_temp.toarray()))
		if args.impl =='qutip':
			H0 = qt.Qobj(H0)
			H1 = qt.Qobj(H1)
			L = [qt.Qobj(Li) for Li in L]

		dyna_type = 'lind_new'
		fid_type = 'GRK'

	elif test_case =='TLSsec_bath_2qb':
		'''Derived from the testcase Lloyd_var1'''
		n1 = args.env_dim #number of bath qubits coupled to qubit 1
		n2 = args.env_dim2 #number of bath qubits coupled to qubit 2
		n_b=n1+n2
		n_system = 2
		if args.cs_coup == 'eq':
			A = np.ones(n1+n2)
			A /= 200
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.5, 5, n1+n2)
			A /= 1000 #Scale to desired strength
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# E = np.linspace(1.0,1.0+0.1*(n_b-1), num=2) #TODO:qubit frequency
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
		
		decoherence_type = args.deco_type
		gamma = [np.sqrt(1/args.T1_sys)]*n_system + [np.sqrt(1/args.T1_TLS)]*n_b 
		gamma = [gamma_i/np.sqrt(8) for gamma_i in gamma] #Strength Scaling

		L = []
		for i in range(n_b+n_system):
			L_term = [[decoherence_type, [[gamma[i], i]] ]]
			H_temp = hamiltonian(L_term, [], basis=basis, dtype=np.complex128, check_herm=False)
			L.append(np.array(H_temp.toarray()))
		if args.impl =='qutip':
			H0 = qt.Qobj(H0)
			H1 = qt.Qobj(H1)
			L = [qt.Qobj(Li) for Li in L]

		dyna_type = 'lind_new'
		fid_type = 'GRK'

	elif test_case == 'Xmon_nb':
		'''
		Rotating frame, no bath energy
		'''
		dyna_type = 'cs'
		fid_type = 'au'
		n_b = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n_b)
			A /= 200
		elif args.cs_coup == 'uneq': #Only consider unequal case, which is the case of real systems
			A = np.random.uniform(0.5, 5, n_b)
			A /= 1000 #Scale to desired strength
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		
		# compute site-coupling lists
		z_term = [[-0.5, 0]]
		couple_term = [[A[i]/2, 0, i+1] for i in range(n_b)]
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
	
	elif test_case == 'dipole_VarStr':
		dyna_type = 'cs'
		fid_type = 'au'
		n_b = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n_b)
		elif args.cs_coup == 'uneq':
			A = np.random.normal(1.0, 0.25, n_b)
		
		A /= 10**(args.cp_str) #sacle to desired strength, power of 10

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		
		# compute site-coupling lists
		z_term = [[-0.5, 0]]
		couple_term = [[A[i], 0, i+1] for i in range(n_b)]
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
	
	elif test_case == 'dipole_VarStr_lab':
		dyna_type = 'cs'
		fid_type = 'au'
		n_b = args.env_dim #number of bath qubits
		if args.cs_coup == 'eq':
			A = np.ones(n_b)
		elif args.cs_coup == 'uneq':
			A = np.random.normal(1.0, 0.25, n_b)
		
		A /= 10**(args.cp_str) #sacle to desired strength, power of 10

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+1)
		
		Delta = np.linspace(1.0,1.0+0.1*(n_b-1), num=n_b) #TLS frequencies
		Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency
		
		# compute site-coupling lists
		z_term = [[-Delta[i]/2, i] for i in range(n_b+1)]
		couple_term = [[A[i], 0, i+1] for i in range(n_b)]
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

	elif test_case == '2qbiSWAP':
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
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# E = np.linspace(1.0,1.0+0.1*(n_b-1), num=2) #TODO:qubit frequency
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

		n_b=n1+n2
		n_system = 2

	elif test_case == '2qbCPHASE':
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
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# E = np.linspace(1.0,1.0+0.1*(n_b-1), num=2) #TODO:qubit frequency
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
		n_b=n1+n2
		n_system = 2
	
	elif test_case == 'Lloyd_2qb':
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
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# E = np.linspace(1.0,1.0+0.1*(n_b-1), num=2) #TODO:qubit frequency
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
		n_b=n1+n2

	elif test_case == 'Lloyd_var1':
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
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# E = np.linspace(1.0,1.0+0.1*(n_b-1), num=2) #TODO:qubit frequency
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
		n_b=n1+n2

	elif test_case == 'Lloyd_3qb':
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
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# E = np.linspace(1.0,1.0+0.1*(n_b-1), num=2) #TODO:qubit frequency
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
		n_b=n1+n2

	elif test_case == 'XXnYY':
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
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# E = np.linspace(1.0,1.0+0.1*(n_b-1), num=2) #TODO:qubit frequency
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
		n_b=n1+n2
		n_system = 2
	
	elif test_case == 'XXYY_X':
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
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		# E = np.linspace(1.0,1.0+0.1*(n_b-1), num=2) #TODO:qubit frequency
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
		n_b=n1+n2
		n_system = 2

	elif test_case == 'XXpm':
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
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0

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
		n_b=n1+n2
		n_system = 2

	elif test_case == '1qb1anci':
		'''Proved to be universal:
		https://arxiv.org/abs/1812.11075v1
		On the universality of the quantum approximate optimization algorithm
		Change to a more practical scenario
		Now treating the second qubit as ancilla
		'''
		dyna_type = 'cs'
		fid_type = 'au'
		n1 = args.env_dim #number of bath qubits coupled to the qubit
		n2 = args.env_dim2 #number of bath qubits coupled to the ancilla
		n_system = 1
		n_b=n1+n2+1 #The ancilla is treated as bath when computing the fidelity
		
		if args.cs_coup == 'eq':
			A = np.ones(n1+n2)
			A /= 200
		elif args.cs_coup == 'uneq':
			A = np.random.uniform(0.5, 5, n1+n2)
			A /= 1000 #Scale to desired strength
		A *= args.cs_coup_scale if hasattr(args,'cs_coup_scale') else 1.0
		
		g = 1.0 #qubit-qubit coupling constant

		# compute Hilbert space basis
		basis = spin_basis_1d(L = n_b+n_system)
		
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

	elif test_case == 'result_analysis':
		'''Use this with caution!!!'''
		dyna_type = 'cs'
		fid_type = 'au'
		n1 = args.env_dim #number of bath qubits coupled to qubit 1
		n2 = args.env_dim2 #number of bath qubits coupled to qubit 2
		n_system = 2
		
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
		n_b=n1+n2

	psi1 =  target_uni(args.au_uni)
	psi1_input = psi1.astype(complex)
	if len(psi1_input) < 2**n_system:
		'''Append identity if dimension doesn't match'''
		Id = np.identity(2**n_system // len(psi1_input))
		psi1_input = np.kron(psi1_input, Id)
		
	psi0_input = np.identity(2**(n_b+n_system)) #Start from identity
	
	if test_case in {'TLSsec_bath', 'TLSsec_bath_2qb', 'TLSsec_bath_lowStr', 'Koch_1qb', 'Koch_paper_1qb'}:
		#note that psi0 here is meaningless
		quma = QuManager(psi0_input, psi1_input, H0, H1, dyna_type, fid_type, args,
						couplings = A, lind_L = L, n_s = n_system, n_b = n_b)
	elif interaction_pic:
		quma = QuManager(psi0_input, psi1_input, H0, H1, dyna_type, fid_type, args, n_s = n_system, n_b = n_b, couplings = A, H_B = H_B, H_int = H_int, H_S0 = H_S0, H_S1 = H_S1)
	elif test_case in {'iso_4Ham','dp_4Ham'}:
		quma = QuManager(psi0_input, psi1_input, H0, H1, dyna_type, fid_type, args, n_s = n_system, n_b = n_b, couplings = A,H2=H2, H3=H3)
	else:
		quma = QuManager(psi0_input, psi1_input, H0, H1, dyna_type, fid_type, args, n_s = n_system, n_b = n_b, couplings = A)
	
	return quma