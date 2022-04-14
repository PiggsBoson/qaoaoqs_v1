import numpy as np
import scipy.linalg as la
from functools import partial
import os
from Dynamics import *
import qutip as qt

#Pauli matrices
sigma_x = np.matrix([[0.0, 1.0], [1.0, 0.0]])
sigma_z = np.matrix([[1.0, 0.0], [0.0, -1.0]])
sigma_y = 1.0j * np.matrix([[0.0, -1.0], [1.0, 0.0]])
'''
To test the effect of negative time on Lindblad
'''

psi0 = np.array([-1. / 2 - np.sqrt(5.) / 2, 1])
psi0 = psi0 / la.norm(psi0)
psi0_input = psi0.astype(complex)

psi1 = np.array([1. / 2 + np.sqrt(5.) / 2, 1])
psi1 = psi1 / la.norm(psi1)
psi1_input = psi1.astype(complex)

sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])

H0 = (-sigma_z + 4.0 * sigma_x) / 2.
H1 = (-sigma_z - 4.0 * sigma_x) / 2.

T = 21.0 #time

#chcek orthogonality
pdt = psi0_input.conj().T @ psi1_input
print(pdt)

# u = np.copy(psi0_input)
# u = np.outer(u, u.conj().T) #convert u to density matrix
# uv = u.reshape((-1, 1), order='F')#vectorize
# simulator = Dyna(uv , H0, u.shape[0], 1.0, sigma_z, 'vec')
# uv = simulator.simulate_lind_vec(T)
# u = uv.reshape(u.shape, order='F')
# result = np.absolute(psi1_input.conj().T @ u @ psi1_input)[0,0] ** 2

#Test the behavior of the result
# uf = np.copy(psi0_input)
# uf = np.outer(uf, uf.conj().T)
# uvf = uf.reshape((-1, 1), order='F')#vectorize
# uf = uvf.reshape(uf.shape, order='F') 
# resultf = np.absolute(psi1_input.conj().T @ uf @ psi1_input) ** 2
# print(resultf)

#Test the behavior of the reshaping
# test1 = np.matrix([[1,2],[3,4]])
# print(test1)
# test1v = test1.reshape((-1, 1), order='F')
# print(test1v)
# test1 = test1v.reshape(test1.shape, order='F')
# print(test1)
