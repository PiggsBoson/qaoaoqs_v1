#Modified with Zhibo‘s version of dynamics
#Date Jan. 14 2020
import numpy as np
import scipy.linalg as la
from qaoaoqs.Dynamics import *


#Pauli matrices
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
sigma_y = 1.0j * np.array([[0.0, -1.0], [1.0, 0.0]])

class QuManager():
	# quantum env for the agent
	def __init__(self, psi0, psi1, H0, H1, dyna_type, fid_type, args = None, couplings = None, H_ns1 = None, H_ns2 = None, **kwargs):
		"""quantum environment for the RL agents 

		Arguments:
			psi0 -- initial quantum state
			psi1 -- target quantum state
			H0 -- the first hamiltonian gate 
			H1 -- the second hamiltonian gate 
			dyna_type -- type of dynamics
				'cs' = Central Spin
				'lind' = lindblad
			fid_type -- type of fidelity definition
				'st'= state transfer, 
				'au' = arbitary unitary
			impl -- implementation method
			

		Keyword Arguments:
			n -- number of bath qubits coupled to qubit 1
			n2 -- number of bath qubits coupled to qubit 2
			fid_fix -- method to fix time negativity
			H_ns1 -- noise hamiltonian gate 1
			H_ns2 -- noise hamiltonian gate 2
			delta -- gaussian noise level
			renormal -- Whether renormalizing the protocol in computing the fidelity
		"""
		self.psi0 = psi0 #Added. Used for density matrix
		self.imag_unit = np.complex64(1.0j)
		self.psi1 = psi1# .reshape(psi1.shape[0], 1) #Change here

		self.H1 = H1
		self.H0 = H0

		# self.H0_eval, self.H0_evec = la.eigh(H0)
		# self.H0_eval = np.expand_dims(self.H0_eval, axis=-1) #This decomposition aids efficiency

		# self.H1_eval, self.H1_evec = la.eigh(H1)
		# self.H1_eval = np.expand_dims(self.H1_eval, axis=-1) #This decomposition aids efficiency

		self.dyna_type = dyna_type
		self.fid_type = fid_type
		
		self.couplings = couplings

		if args: 
			self.impl = args.impl
			self.n = args.env_dim
			if hasattr(args,'env_dim2'):
				self.n2 = args.env_dim2
			else:
				self.n2 = 0

			self.fid_fix = args.fid_fix
			self.fix_adj = args.fid_adj
			if hasattr(args,'protocol_renormal'):
				self.renormal = args.protocol_renormal
			else:
				self.renormal = False
			if hasattr(args,'T_tot'):
				self.T_tot = args.T_tot
			
			self.testcase = args.testcase
			self.return_uni = False
			
			if hasattr(args,'ode_steps'):
				self.ode_steps = args.ode_steps

		else:
		#enables convenient implementation, especially for testing
			self.impl = kwargs.get('impl', None)
			self.n = kwargs.get('env_dim')
			self.n2 = kwargs.get('env_dim2',0)
		
			self.renormal = kwargs.get('protocol_renormal', False)
			self.fid_fix = kwargs.get('fid_fix', None)
			self.fix_adj = kwargs.get('fix_adj', None)
			self.T_tot = kwargs.get('T_tot',0)
			

			self.delta = kwargs.get('noise_delta',0)

			self.testcase = kwargs.get('testcase')

			self.return_uni = kwargs.get('return_uni', False)

		
		if self.dyna_type == 'lind_new':
			self.simulator = Dyna(psi0, H0, N=2**(self.n+1), L = kwargs.get('lind_L',None), impl = self.impl, t_steps = args.ode_steps)
	
	def get_reward(self, protocol):
		"""Get the fidelity of the protocol

		Arguments:
			protocol -- The alpha's and beta's for a given protocol

		Returns:
			fildeity -- scalar between 0 and 1
		"""
		
		result = 0

		if self.renormal: 
			#checked: pass by value here. Will not afffect the value of the protocol
			protocol /= np.sum(protocol)
			protocol *= self.T_tot
		if self.fid_fix =='abs':
			protocol = np.abs(protocol)

		if self.dyna_type == 'cs':
			u = np.copy(self.psi0)
			simulator = Dyna(u , self.H0)
			for i in range(len(protocol)):
				if i % 2 == 0:
					simulator.setH(self.H0)
					u = simulator.simulate_closed(protocol[i])
				else:
					simulator.setH(self.H1)
					u = simulator.simulate_closed(protocol[i])
			if self.return_uni:
				return u
			
			if self.n ==0:
				#No bath
				N_s = self.H0.shape[0]
				tgt = np.matrix(self.psi1)
				result = np.absolute(np.trace(tgt.getH() @ u) / N_s) ** 2
			else:
				#The fidelity follows that defined by Arenz et al.
				n_bath = self.n + self.n2 #total number of bath qubits
				N_b = 2**n_bath #size of bath space
				N_s = int(self.H0.shape[0]/N_b) #size of system space
				u_f = np.kron(self.psi1, np.identity(N_b))
				Q = np.matmul(u_f.conjugate().transpose(), u )
				#Implementation 1
				# Q = Q[ :N, :N] + Q[N: , N: ] #partial trace
				#Implementation 2: https://www.peijun.me/reduced-density-matrix-and-partial-trace.html
				Q_tensor = Q.reshape([N_s, N_b, N_s, N_b])
				Q_b = np.trace(Q_tensor, axis1=0, axis2=2) #partial trace over sys
				result = np.absolute(np.trace(la.sqrtm(Q_b.conjugate().transpose() @ Q_b)) / (N_s*N_b)) ** 2
				if self.fid_fix =='barrier':
					result = result - 100000000.0 * int(np.any([t<0 for t in protocol]))

		elif self.dyna_type == 'lind_new':
			for i in range(len(protocol)):
				if i % 2 == 0:
					self.simulator.setH(self.H0)
					self.simulator.simulate_lind_qutip(protocol[i])
				else:
					self.simulator.setH(self.H1)
					self.simulator.simulate_lind_qutip(protocol[i])
							
		if self.fix_adj =='t':
			# if np.sum(protocol) >= 1:
			result -= np.log(np.sum(protocol))
		return result

	def state_fidelity(self, protocol):
		"""Get the fidelity with a given initial state a target state only for the system. The initial and final states are all density matricies.

		Arguments:
			protocol -- The alpha's and beta's for a given protocol

		Returns:
			fildeity -- scalar between 0 and 1
		"""
		if self.renormal: 
			#checked: pass by value here. Will not afffect the value of the protocol
			protocol /= np.sum(protocol)
			protocol *= self.T_tot
		if self.fid_fix =='abs':
			protocol = np.abs(protocol)
		rhot = np.copy(self.psi0)
		simulator = Dyna(rhot, self.H0)
		for i in range(len(protocol)):
			if i % 2 == 0:
				simulator.setH(self.H0)
				simulator.simulate_closed_rho(protocol[i])
			else:
				simulator.setH(self.H1)
				simulator.simulate_closed_rho(protocol[i])
		rhot = simulator.state
		Nb = 2**(self.n+self.n2)
		rho_s = self.ptrace(rhot, Nb, out = 'b')
		#Changed to ptrace fidelity and overlap with the target state for fidelity check. NOT USED FOR Optimization!!!
		return np.trace(rho_s @ self.psi1).real #the target is a pure state so we can do this

	def without_ctr(self, T):
		"""Get the fidelity without control
		"""
		result = 0
		u = np.copy(self.psi0)
		simulator = Dyna(u , self.H0)
		u = simulator.simulate_closed(T)
		#The fidelity follows that defined by Arenz et al.
		N = 2**self.n #size of bath space
		u_f = np.matrix(np.kron(self.psi1, np.identity(N)))
		Q = u_f.getH() @ u 
		Q = Q[ :N, :N] + Q[N: , N: ] #partial trace
		result = np.absolute(np.trace(la.sqrtm(Q.getH() @ Q)) / (2*N)) ** 2
		return result
	
	def reset(self, **kwargs):
		"""Reset some of the parameters when needed
		"""
		for param, value in kwargs.items():
			self.__dict__[param] = value #This way one can change the class attribute with the name str(param) without repetitive coding
			#Then do some updates
			# if param == 'psi0':
			# 	self.psi0_input = np.expand_dims(self.psi0, axis=1)
			# elif param == 'H0':
			# 	self.H0_eval, self.H0_evec = la.eigh(self.H0)
			# 	self.H0_eval = np.expand_dims(self.H0_eval, axis=-1) #This decomposition aids efficiency
			# elif param == 'H1':
			# 	self.H1_eval, self.H1_evec = la.eigh(self.H1)
			# 	self.H1_eval = np.expand_dims(self.H1_eval, axis=-1) #This decomposition aids efficiency
		
	def ptrace(self, M, nb, out='b'):
		'''partial trace
		https://www.peijun.me/reduced-density-matrix-and-partial-trace.html'''
		ns = len(M)//nb
		M_tensor = M.reshape([ns, nb, ns, nb])
		if out =='b':
			result = np.trace(M_tensor, axis1=1, axis2=3)
		elif out == 's':
			result = np.trace(M_tensor, axis1=0, axis2=2)
		return result
	
	# def get_reward_ns1_m(self, protocol, ns):
	# 	"""Get the fidelity of the type-given protocol in the hamiltonian noise setting I

	# 	Arguments:
	# 		protocol {list} -- protocol
	# 		ns {scalar} -- the number of noise to sample 

	# 	Returns:
	# 		sampled mean fildelity, sampled min fidelity -- scalar between 0 and 1 
	# 	"""
	# 	delta = self.delta
	# 	l_w1 = np.random.uniform(low=-delta, high=delta, size=ns)
	# 	l_w2 = np.random.uniform(low=-delta, high=delta, size=ns)

	# 	l_fid = []

	# 	for (w1, w2) in zip(l_w1, l_w2):
	# 		H0_eval, H0_evec = la.eigh(self.H0 - w1*self.H_ns1 - w2*self.H_ns2)
	# 		H0_eval = np.expand_dims(H0_eval, axis=-1)
	# 		H1_eval, H1_evec = la.eigh(self.H1 - w1*self.H_ns1 - w2*self.H_ns2)
	# 		H1_eval = np.expand_dims(H1_eval, axis=-1)

	# 		u = np.copy(self.psi0_input)
	# 		for i in range(len(protocol)):
	# 			if i % 2 == 0:
	# 				np.matmul(H0_evec.conj().T, u, out=u)
	# 				np.multiply(
	# 					np.exp(-protocol[i] * self.imag_unit * H0_eval), u, out=u)
	# 				np.matmul(H0_evec, u, out=u)
	# 			else:
	# 				np.matmul(H1_evec.conj().T, u, out=u)
	# 				np.multiply(
	# 					np.exp(-protocol[i] * self.imag_unit * H1_eval), u, out=u)
	# 				np.matmul(H1_evec, u, out=u)
	# 		l_fid.append(np.absolute(
	# 			np.dot(self.psi1.T.conjugate(), u))[0][0] ** 2)

	# 	return np.mean(l_fid), np.min(l_fid)

	# def get_reward_ns2_m(self, protocol, ns):
	# 	"""Get the fidelity of the type-given protocol in the hamiltonian noise setting II

	# 	Arguments:
	# 		protocol {list} -- protocol
	# 		ns {scalar} -- the number of noise to sample 

	# 	Returns:
	# 		sampled mean fildelity, sampled min fidelity -- scalar between 0 and 1 
	# 	"""
	# 	delta = self.delta
	# 	l_w1 = np.random.uniform(low=0.0, high=delta, size=ns)

	# 	l_fid = []

	# 	for w1 in l_w1:
	# 		H0_eval, H0_evec = la.eigh(self.H0 + w1*self.H_ns1 )
	# 		H0_eval = np.expand_dims(H0_eval, axis=-1)

	# 		H1_eval, H1_evec = la.eigh(self.H1)
	# 		H1_eval = np.expand_dims(H1_eval, axis=-1)

	# 		u = np.copy(self.psi0_input)
	# 		for i in range(len(protocol)):
	# 			if i % 2 == 0:
	# 				np.matmul(H0_evec.conj().T, u, out=u)
	# 				np.multiply(
	# 					np.exp(-protocol[i] * self.imag_unit * H0_eval), u, out=u)
	# 				np.matmul(H0_evec, u, out=u)
	# 			else:
	# 				np.matmul(H1_evec.conj().T, u, out=u)
	# 				np.multiply(
	# 					np.exp(-protocol[i] * self.imag_unit * H1_eval), u, out=u)
	# 				np.matmul(H1_evec, u, out=u)

	# 		l_fid.append(np.absolute(
	# 			np.dot(self.psi1.T.conjugate(), u))[0][0] ** 2)

	# 	return np.mean(l_fid), np.min(l_fid)

	# def get_reward_noise(self, noise_level, protocol, batchsize=1):
	# 	"""Adding the gaussian noise to the fidelity

	# 	Arguments:
	# 		noise_level {scalar} -- std for the gaussian 
	# 		protocol {list} -- protocol 
	# 		batchsize {scalar} -- batchsize to be average on; if equal to 1, no averaging.

	# 	Returns:
	# 		fildeity -- noised fidelity with the guassian noise
	# 	"""
	# 	fid = self.get_reward(protocol)
	# 	noise = np.random.normal(0.0, noise_level, batchsize)
	# 	fid = np.clip(fid + noise, 0.0, 1.0)
	# 	return np.mean(fid)

	# def get_reward_quantum(self, protocol, batchsize=1):
		# """quantum binary measurement of the fidelity

		# Arguments:
		# 	protocol {list} -- protocol 
		# 	batchsize {scalar} -- batchsize to be average on; if equal to 1, no averaging.


		# Returns:
		# 	fildeity -- binary noised fidelity 
		# """
		# fid = self.get_reward(protocol)
		# fid = np.random.binomial(1, p=fid, size=batchsize)
		# return np.mean(fid)
