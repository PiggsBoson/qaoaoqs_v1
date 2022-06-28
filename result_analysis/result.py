from qaoaoqs.Dynamics import *
import qaoaoqs.sys_setup as sys_setup
from qaoaoqs.quantum_manager import *
import qutip as qt
"""
For convenient result analysis including plotting, storing (not implemented yet)
"""
zero_z = np.array([1.0,0.0]).astype('complex128')
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]]).astype('complex128')

from scipy.stats import rv_continuous
class sin_prob_dist(rv_continuous):
	'''https://pennylane.ai/qml/demos/tutorial_haar_measure.html'''
	def _pdf(self, theta):
		# The 0.5 is so that the distribution is normalized
		return 0.5 * np.sin(theta)

class Result():
	def __init__(self, exp_name, data, params):
		"""system parameters

		Arguments:
			exp_name -- the name of the simulation
			data -- results
			params -- parameters
		"""
		self.exp_name = exp_name
		self.data = data
		self.params = params
		if params.testcase == 'TLSsec_bath':
			self.data["T_tot"] *= 8


	def naive_rd_state(self,qubit_ct=1):
		'''This one will give a arc on Bloch sphere'''
		sys_init = np.random.rand(2**qubit_ct).astype('complex128') #random initial state of bath
		sys_init /= np.sqrt(sys_init.conj().T @ sys_init) #Normalize
		return np.outer(sys_init, sys_init.conj().T)

	def Haar_rd_state(self, qubit_ct):
		'''
		Generating a single qubit Haar random state.
		https://pennylane.ai/qml/demos/tutorial_haar_measure.html
		uniformly random phi and sin random theta
		'''
		if qubit_ct ==1:
			phi = 2 * np.pi * np.random.uniform()
			# Samples of theta should be drawn from between 0 and pi
			sin_sampler = sin_prob_dist(a=0, b=np.pi)
			theta = sin_sampler.rvs()
			psi = np.array([np.cos(theta/2), np.exp(-1.0j*phi)*np.sin(theta/2)])
		else:
			#We can just use qutip but I don't want to waste the codes already written...
			psi = qt.rand_ket_haar(N=2**qubit_ct)
			psi = np.array(psi.full())
		return np.outer(psi, psi.conj().T)

	def state_fid(self, temp = 'zero', no_samples=100):
		"""Checking how the protocol gained from unitary fidelity works for a specific state. The dynamics should be mainly implemented in quantum_manager
		Idea: Choosing Haar-random qubit state and randomized bath states. Use overlap between the final density matrix and the target. Use several random bath states as the initial states and make an error bar. 
		
		Arguments:
			qubit_ct -- number of system qubit
			temp -- bath temperature
		"""
		if hasattr(self.params, 'couplings'):
			quma = sys_setup.setup(self.params, couplings = self.params.couplings)
		else:
			quma = sys_setup.setup(self.params)
		qubit_ct = int(np.log2(len(quma.psi1)))
		target_uni = quma.psi1
		outputs = []
		n = quma.n+quma.n2
		if temp == 'zero':
			bath_init = np.zeros((2**n, 2**n), dtype='complex128')
			bath_init[0,0] = 1
		elif temp == 'inf':
			bath_init = np.identity(2**n, dtype='complex128') #unpolarized state of bath
			bath_init /= 2**n #Normalize
		else:
			bath_init = self.Haar_rd_state(n)
		for _ in range(no_samples):
			sys_init = self.Haar_rd_state(qubit_ct)

			quma.reset(psi0 = np.kron(sys_init,bath_init))
			quma.reset(psi1 = target_uni @ sys_init @ target_uni.conj().T)
			outputs.append(quma.state_fidelity(self.data['duration'].to_numpy()[0][0]))
		log_out = [-np.log10(1 - f) for f in outputs] #Just keep the log
		# print("The mean fidelity is", np.mean(outputs))
		# print("The mean log fidelity is", np.mean(log_out))
		# print("The std of fidelity is", np.std(outputs))
		# print("The std of log fidelity is", np.std(log_out))
		self.data.insert(len(self.data.columns), 'state_fid_log', np.mean(log_out))
		self.data.insert(len(self.data.columns), 'state_fid_log_sd', np.std(log_out)) #This std is different from the real std, but should be good for now. 


	def __getHam__(self, uni, T):
		'''
		Given a target unitary U and total time T, return the corresponding Hamiltonian H. i.e. exp(iHT) = U
		Idea: use diagonalization. https://physics.stackexchange.com/questions/460407/how-does-a-hamiltonian-generate-a-unitary (I also came up with this myself)
		'''
		#TODO: not really useful to implement it fully for general unitary for now
		pass

	def no_opt(self):
		'''The fidelity without control'''
		quma = sys_setup.setup(self.params)
		# target = sys_setup.target_uni(self.params.au_uni)
		# quma.reset(psi1 = target)
		#TODO: WARNING!!! It only works for -z now!!!
		from quspin.basis import spin_basis_1d
		from quspin.operators import hamiltonian
		# compute Hilbert space basis
		n = self.params.env_dim
		basis = spin_basis_1d(L = n+1)
		if self.params.cs_coup == 'eq':
			A = np.ones(n)
		# compute site-coupling lists
		z_term = [[-1.0/self.params.T_tot*np.pi/2, 0]] #This is only for -z target
		couple_term = [[A[i], 0, i+1] for i in range(n)]

		#operator string lists
		static_d = [['z', z_term], ['zz', couple_term], 
					['xx', couple_term], ['yy', couple_term]]

		H = hamiltonian(static_d, [], basis=basis, dtype=np.complex128).toarray() 
		quma.reset(H0=H)
		self.data.insert(len(self.data.columns), 'no_opt', quma.without_ctr(self.params.T_tot))
		self.data.insert(len(self.data.columns), 'log_fid_no', -np.log10(1 - self.data["no_opt"]) )
		# compute site-coupling lists
		# n = data['env_dim'].to_numpy()[0] #number of bath qubits
		# T = data['T_tot'].to_numpy()[0] #Total duration
		
		# A = np.full(n, 1.0) #coupling constants. Set to be equal here
		# H =  CentralSpin.Hamiltonian( -1.0/T*np.pi/2*sigma_z, A)
		# u = np.identity(2**(n+1))
		# simulator = Dyna(u ,H)
		# u = simulator.simulate_closed(T)
		# #The fidelity follows that defined by Arenz et al.
		# N = 2**n #size of bath space
		# u_f = np.kron(psi1, np.identity(N))
		# Q = u_f.conj().transpose() @ u 
		# Q = Q[ :N, :N] + Q[N: , N: ] #partial trace
		# result = np.absolute(np.trace(la.sqrtm(Q.conj().transpose() @ Q)) / (2*N)) ** 2
		
	def log_fid(self):
		self.data.insert(len(self.data.columns), 'log_fid', -np.log10(1 - self.data["fid"]) )

	def transmon_time(self):
		self.data.insert(len(self.data.columns), 'TLS_T', self.data["T_tot"] / (8 *2*np.pi))

	def transmon_time_1G(self):
		self.data.insert(len(self.data.columns), 'TLS_T', self.data["T_tot"] / (2*np.pi))

	def no_bath(self):
		'''Checking the implemented unitary without bath using a protocol optimized in the presence of bath.'''
		quma = sys_setup.setup(self.params, if_no_bath = True)
		quma.reset(n=0)
		result  = quma.get_reward(self.data['duration'].to_numpy()[0][0])
		self.data.insert(len(self.data.columns), 'no_bath_fid', result )
		self.data.insert(len(self.data.columns), 'no_bath_fid_log', -np.log10(1-result) )

	def QAOA_uni(self):
		'''Return the final unitary as a result of QAOA evolution'''
		# self.params.testcase = 'result_analysis'
		quma = sys_setup.setup(self.params, couplings = self.data['couplings'].to_numpy()[0])
		quma.reset(return_uni=True)
		return quma.get_reward(self.data['duration'].to_numpy()[0][0])

	def compute_fid(self):
		'''recompute fidelity'''
		# self.params.testcase = 'result_analysis'
		quma = sys_setup.setup(self.params, couplings = self.data['couplings'].to_numpy()[0])
		return quma.get_reward(self.data['duration'].to_numpy()[0][0])

	def HA_fraction(self):
		'''The Fraction of time applying H_A'''
		self.data.insert(len(self.data.columns), 'HA_frac', sum(self.data['duration'].to_numpy()[0][0][0::2])/self.params.T_tot )

	def GRK_fidelity(self, temp = 'zero',):
		'''Optimal control theory for a unitary operation under dissipative evolution'''
		quma = sys_setup.setup(self.params)
		N_sys = len(quma.psi1)

		# w= [20/22, 1/22, 1/22]
		# rho1 = np.zeros(quma.psi1.shape)
		# for i in range(N_sys):
		# 	rho1[i,i] = 2*(N_sys-i)/(N_sys*(N_sys+1))
		# rho_test = [rho1,
		# 			np.ones(quma.psi1.shape)/N_sys,
		# 			np.identity(N_sys)/N_sys]
		rho_test = []
		w= np.ones(N_sys+1)/(N_sys+1)
		for i in range(N_sys):
			rhoi = np.zeros(quma.psi1.shape)
			rhoi[i,i] = 1
			rho_test.append(rhoi)
		rho_test.append(np.ones(quma.psi1.shape)/N_sys)


		target_uni = quma.psi1
		outputs = []
		n = quma.n+quma.n2
		if temp == 'zero':
			bath_init = np.zeros((2**n, 2**n), dtype='complex128')
			bath_init[0,0] = 1
		elif temp == 'inf':
			bath_init = np.identity(2**n, dtype='complex128') #unpolarized state of bath
			bath_init /= 2**n #Normalize
		else:
			bath_init = self.Haar_rd_state(n)
		for i in range(len(rho_test)):
			rhoi = rho_test[i]
			quma.reset(psi0 = np.kron(rhoi,bath_init))
			quma.reset(psi1 = target_uni @ rhoi @ target_uni.conj().T)
			outputs.append(np.real(quma.state_fidelity(self.data['duration'].to_numpy()[0][0]))* w[i]/ np.trace(rhoi@rhoi))
		result = np.sum(outputs)
		print(result)
		self.data.insert(len(self.data.columns), 'GRK_fid_log', -np.log10(1 - result) )
