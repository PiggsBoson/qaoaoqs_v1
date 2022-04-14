'''Benchmarking run time of different implementations of lindblad dynamics
options: numpy superoperator, qutip, quspin? np QT'''
import time
import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
import matplotlib.pyplot as plt

from qaoaoqs.Dynamics import *
from qaoaoqs.tools import *

n_system = 1
n = 3
 #number of bath qubits
N = 2**(n+n_system)
print("number of bath spins: ", n)
A = np.ones(n)
A *= 0.04 #40MHz
omega = 8.0 #GHz

sys_init = np.array([[1.0,0.0], [0.0,0.0]])
bath_init = np.zeros((2**n, 2**n), dtype='complex128')
bath_init[0,0] = 1
rho0  = np.kron(sys_init,bath_init)
Delta = np.linspace(1.0,1.0+0.1*(n-1), num=n) #TLS frequencies
Delta = np.insert(Delta, 0, 1.0) #Insert the qubit frequency
Delta *= omega
# compute Hilbert space basis
basis = spin_basis_1d(L = n+1)
# compute site-coupling lists
z_term = [[-Delta[i]/2, i] for i in range(n+1)]
couple_term = [[A[i]/2, 0, i+1] for i in range(n)]
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

decoherence_type = '-'
gamma = [np.sqrt(1/5)] + [np.sqrt(1/1)]*n #sqrt(1/T1)
L = []
for i in range(n+n_system):
    L_term = [[decoherence_type, [[gamma[i], i]] ]]
    H_temp = hamiltonian(L_term, [], basis=basis, dtype=np.complex128, check_herm=False)
    L.append(np.array(H_temp.toarray()))

obs = sigma_x
obs = np.kron(obs, np.identity(2**n))
rng = np.random.default_rng(1984) #Random number generator
p = 30 #contorl depth
protocol = rng.random(size = p*2)#generate random protocol
protocol *= 1
tarray = np.cumsum(protocol)
repeats = 1
time_steps = 2

# print(rho0)

'''Method 0: unitary dynamics for comparison'''
simulator0 = Dyna(rho0, H0, N)
t_tot = 0
for _ in range(repeats):
    measured0 = []
    start = time.time()
    for i in range(len(protocol)):
        if i % 2 == 0:
            simulator0.setH(H0)
            simulator0.simulate_closed_rho(protocol[i])
            measured0.append(simulator0.measure(obs))
        else:
            simulator0.setH(H1)
            simulator0.simulate_closed_rho(protocol[i])
            measured0.append(simulator0.measure(obs))
    t_tot += time.time() - start
    plt.plot(tarray, measured0, marker = '1',label = '0')
print("Method 0: unitary dynamics for comparison: ", t_tot/repeats)

'''Method 1 numpy with vectorization'''
start = time.time()
simulator1 = Dyna(rho0, H0, N, L, lind_vec = True)
measured1 = []
for i in range(len(protocol)):
    if i % 2 == 0:
        simulator1.setH(H0)
        simulator1.simulate_lind_vec(protocol[i])
        measured1.append(simulator1.measure(obs))
    else:
        simulator1.setH(H1)
        simulator1.simulate_lind_vec(protocol[i])
        measured1.append(simulator1.measure(obs))
plt.plot(tarray, measured1, marker = '2',label = '1')
print("Method 1 numpy with vectorization: ", time.time() - start)

# """Method 2: numpy ode: """
# start = time.time()
# simulator2 = Dyna(rho0, H0, N, L, t_steps = time_steps)
# # measured2 = []
# for i in range(len(protocol)):
#     if i % 2 == 0:
#         simulator2.setH(H0)
#         simulator2.simulate_lind_npode(protocol[i])
#         # measured2.append(simulator2.measure(obs))
#     else:
#         simulator2.setH(H1)
#         simulator2.simulate_lind_npode(protocol[i])
#         # measured2.append(simulator2.measure(obs))
# # plt.plot(tarray, measured2, marker = '3',label = '2')
# print("Method 2: numpy ode: ", time.time() - start)

"""Method 3: qutip: """
start = time.time()
simulator3 = Dyna(rho0, H0, N=N, L = L, impl = 'qutip', t_steps = time_steps)
t_tot = 0
for _ in range(repeats):
    measured3= []
    start = time.time()
    for i in range(len(protocol)):
        if i % 2 == 0:
            simulator3.setH(H0)
            simulator3.simulate_lind_qutip(protocol[i])
            measured3.append(simulator3.measure(obs))
        else:
            simulator3.setH(H1)
            simulator3.simulate_lind_qutip(protocol[i])
            measured3.append(simulator3.measure(obs))
    t_tot += time.time() - start
    plt.plot(tarray, measured3, marker = '4',label = '3')
print("Method 3: qutip: ", t_tot/repeats)

# """Method 4: quspin: """
# start = time.time()
# simulator4 = Dyna(rho0, H0, N=N, L = L, impl = 'qutip')
# t_tot = 0
# for _ in range(repeats):
#     measured4= []
#     start = time.time()
#     for i in range(len(protocol)):
#         if i % 2 == 0:
#             simulator4.setH(H0)
#             simulator4.simulate_lind_qutip(protocol[i])
#             measured4.append(simulator4.measure(obs))
#         else:
#             simulator4.setH(H1)
#             simulator4.simulate_lind_qutip(protocol[i])
#             measured4.append(simulator4.measure(obs))
#     t_tot += time.time() - start
#     plt.plot(tarray, measured4, marker = '4',label = '4')
# print("Method 4: quspin: ", t_tot/repeats)

plt.legend()
plt.show()
"""
number of bath spins:  1
Method 0: unitary dynamics for comparison:  0.046922922134399414
Method 1 numpy with vectorization:  0.0637671947479248
Method 2: numpy ode:  1.5916249752044678
Method 3: qutip:  0.4054644823074341

number of bath spins:  2
Method 0: unitary dynamics for comparison:  0.026817798614501953
Method 1 numpy with vectorization:  0.278156042098999
Method 2: numpy ode:  2.077683925628662
Method 3: qutip:  0.5400867462158203

number of bath spins:  3
Method 0: unitary dynamics for comparison:  0.02422299385070801
Method 1 numpy with vectorization:  4.19312310218811
Method 2: numpy ode:  3.125500202178955
Method 3: qutip:  0.7155057668685914

number of bath spins:  4
Method 0: unitary dynamics for comparison:  0.03640482425689697
Method 1 numpy with vectorization:  81.54041790962219
Method 2: numpy ode:  10.018388986587524
Method 3: qutip:  2.1129989862442016

number of bath spins:  5
Method 0: unitary dynamics for comparison:  0.09250009059906006
Method 3: qutip:  4.948623609542847

number of bath spins:  6
Method 0: unitary dynamics for comparison:  1.1378232717514039

number of bath spins:  7
Method 0: unitary dynamics for comparison:  1.9191678524017335
"""
