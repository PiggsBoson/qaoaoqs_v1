#-------------------------------------------------------------------------
# Quantum Dynamics simulation 
#
#-------------------------------------------------------------------------
#
# Author: Z. Yang
# Version 1.0.0
#
#-------------------------------------------------------------------------
import numpy as np
import scipy.linalg as la
from os import path
import sys
from scipy.integrate import solve_ivp
from scipy.integrate import ode
import qutip as qt
from scipy.special import comb

class Dyna():

	def __init__(self, init_state, H, N = 2, L =None, impl = 'numpy', lind_vec =False, t_steps = 100):
		"""system parameters

		Arguments:
			psi0 -- the initial state vector or density matrix
			H -- the Hamiltonian 

		Keyword Arguments:
			N -- Dimentionality of the full system
			impl -- implementation of the dynamics 
			gamma -- the dephasing rate of the lindblad
			A -- the dephasing operator
		"""
		self.N = N
		self.impl = impl
		self.imag_unit = np.complex64(1.0j)
		self.lind_vec = lind_vec
		self.t_steps = t_steps
		if lind_vec:
			self.ori_shape = init_state.shape
			self.state = init_state.reshape((-1, 1), order='F')#vectorize
			self.Id = np.identity(self.N) #identity
			self.H_sup = -1* self.imag_unit *(np.kron(H, self.Id) -np.kron(self.Id, H))
			self.G = 0
			for Li in L:
				self.G += np.kron(Li.conj(), Li) - 0.5* np.kron(self.Id, (Li.conj().T @ Li)) - 0.5 * np.kron((Li.T @ Li.conj()), self.Id)
		else:
			self.state = init_state
			self.H = H
			self.L = L
		
	def setH(self, H):
		if self.lind_vec:
			self.H_sup = - self.imag_unit *(np.kron(H, self.Id) -np.kron(self.Id, H))
		else:
			self.H = H

	def set_state(self, state):
		self.state = state

	def simulate_closed (self, t):
		"""simulate the dynamics of closed system
			implemented with scipy
		"""
		U = la.expm(-1j * self.H * t)
		self.state = U @ self.state

	def simulate_closed_rho (self, t):
		"""simulate the dynamics of closed system with density mtx
			implemented with scipy
		"""
		U = la.expm(-1j * self.H * t)
		self.state = U @ self.state @ U.conj().T
	
	def simulate_lind_vec (self, t):
		"""simulate the lindblad dynamics with vectorized operator approach
		Arguments:
			t -- time for propagation
		"""
		S = la.expm(t*(self.H_sup+self.G))
		self.state = S @ self.state

	def simulate_lind_npode (self, t):
		"""simulate the lindblad dynamics using numpy ode solver. Ineffecitve, not really used.
		Arguments:
			t -- time for propagation
		"""
		rho0 = np.reshape(self.state, self.N**2)
		#rhot = solve_ivp(lambda t, rho: self.__lind(t, rho, self.H, self.gamma, self.A, self.N), [0, t], rho0, method = 'BDF').y[:,-1]
		#rhot = np.reshape(rhot, (self.N, self.N))
		def lindME(t, rho, H, L, N):
			"""The lindblad master equation
			Arguments:
				t -- time for propagation
			"""
			rho = np.reshape(rho, (N, N))
			result = -1.0j*(H@rho - rho@H)
			for Li in L:
				result += (Li @ rho @ Li.conj().T - 1/2* Li.conj().T @ Li @ rho - 1/2*rho @ Li.conj().T @ Li)
			result = np.reshape(result, N**2)
			return result
		r = ode(lindME).set_integrator('zvode', method='bdf').set_initial_value(rho0, 0).set_f_params(self.H, self.L, self.N)
		dt = t/self.t_steps
		while r.successful() and r.t < t:
			r.integrate(r.t+dt)
		rhot = np.reshape(r.y, (self.N, self.N))
		self.state = rhot

	def simulate_lind_qutip (self, t):
		"""simulate the lindblad dynamics using qutip
		Arguments:
			t -- time for propagation
		"""
		self.state = qt.mesolve(self.H, self.state, np.linspace(0,t, num = self.t_steps), self.L).states[-1]

	def measure(self, O):
		'''Measure the current state with observable O'''
		if self.lind_vec:
			return np.trace(O @ self.state.reshape((self.ori_shape), order='F'))
		elif self.impl == 'qutip':
			return qt.expect(qt.Qobj(O), self.state)
		else:
			return np.trace(O @ self.state)
	
	def getRho(self):
		if self.lind_vec:
			return self.state.reshape(self.ori_shape, order='F')
		elif self.impl == 'qutip':
			return self.state.full()
		else:
			return self.state