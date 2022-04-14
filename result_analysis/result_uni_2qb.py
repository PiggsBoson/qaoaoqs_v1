'''
Analysis of the resultant unitary for the 2-qubit cases without bath qubits 
'''
import analysis
from tools import *


result = analysis.get_datasets('2_qubit/Lloyd/PG_Llyod_n0n0_p30T100')
U = result.QAOA_uni()
print('Result Unitary')
print(U)
# print(local_invariants(U))
M_plot(U)
print(result.compute_fid())
print(result.data['fid'].to_numpy()[0])
print(local_invariants(U))
# M_plot(MdgM(U))
print(fidelity(CNOT,U,4,0))
print(fidelity(iSWAP,U,4,1))
'''Test: traced unitary doesn't work at all'''
# psi_i = np.random.rand(8).astype('complex128') #random initial state of bath
# psi_i /= np.sqrt(psi_i.conj().T @ psi_i) #Normalize
# rho_i = np.outer(psi_i, psi_i.conj().T)
# result1 = ptrace(U@rho_i@U.conj().T, 4, 2, out='b' )
# result2 = ptrace(U, 4, 2, out='b' )@ptrace(rho_i, 4, 2, out='b' )@ptrace(U.conj().T, 4, 2, out='b' )
# M_plot(result1-result2)

# U_s = ptrace(U, 4,2,out = 'b')
# # M_plot(MdgM(U_s))
# '''Follwing P360 of Nielsen and Chuang, Kraus operators'''
# E_0 = U_Sys(U, 2, [0,1])
# E_1 = U_Sys(U,2,[1,1])
# M_plot(MdgM(E_0))

'''TODO: test the fidelity measure with: 1. a 2-qubit gate with itself; 2. 2 locally-equivalent gates'''
# U1 = np.kron(CNOT, CNOT)
# print(fidelity(CNOT, U1, 4,4)) #Sanity check for fidelity measure



# M_plot(ptrace(U,4,2,out='b') - SWAP@ptrace(U_swapped, 2,4, out='s')@SWAP)#sanity check, passed!!!
'''Print parameters of target gates'''
# print('CNOT:')
# print(np.linalg.det(CNOT))
# print(local_invariants(CNOT))
# print(local_invariants_angles(CNOT))


# print('iSWAP:')
# print(np.linalg.det(iSWAP))
# print(local_invariants(iSWAP))
# print(local_invariants_angles(iSWAP))

# print('CPHASE:')
# print(local_invariants(CPHASE))

'''Quantum Shannon Decomposition/ Cosine-Sine Decomposition'''
# (B1,B2), theta, (A1,A2) = la.cossin(U_swapped,p=4,q=4, separate=True)
# # B, cs, A = cossin(U,p=4,q=4)

# print('conditionals:')
# print([local_invariants(M) for M in [B1, B2, A1, A2]])
'''so far: the 4 controlled 2-qubit gates are roughly locally equivlant to iSWAP. TODO:How about the CCRy?'''
# # print(cs)
# # M_plot(MdgM(B))