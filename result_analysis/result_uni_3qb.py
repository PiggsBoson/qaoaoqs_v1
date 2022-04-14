'''
Analysis of the resultant unitary for the 2-qubit cases. 
'''
import analysis
from tools import *


'''Matrices'''
SWAP13 = np.diag(np.ones(8))
SWAP13[1][1] = SWAP13[4][4] = SWAP13[3][3] = SWAP13[6][6] = 0
SWAP13[1][4] = SWAP13[4][1] = SWAP13[3][6] = SWAP13[6][3] = 1

SWAP = np.diag(np.ones(4))
SWAP[1][1] = SWAP[2][2] = 0
SWAP[1][2] = SWAP[2][1] = 1

CNOT = np.array([[1, 0, 0, 0], 
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]])

iSWAP = np.array([[1, 0, 0, 0], 
                [0, 0, -1.0j, 0],
                [0, -1.0j, 0, 0],
                [0, 0, 0, 1]])
#Sanity checks    
# testU = np.arange(64).reshape(8,8)
# print(testU)
# testU_tensor = testU.reshape([4, 2, 4, 2])
# print(testU_tensor)
# print(testU_tensor[1,0,0,0])
# print(testU_tensor[0,1,0,0])
# print(testU_tensor[0,0,1,0])
# print(testU_tensor[0,0,0,1])
# testU_s = np.trace(testU_tensor, axis1=1, axis2=3)
# print(testU_s)

# result = analysis.get_datasets('2_qubit/XXnYY/weak2_prPG_XXnYY_n1n0_p15')
result = analysis.get_datasets('2_qubit/XXpm/HeisPG_XXpm_n0n0_p30T50')
U = result.QAOA_uni()
'''swapping qubit 1 and 3 for decomposition'''
# U_swapped  = SWAP13 @U @SWAP13
# print(result.compute_fid())
# print(result.data['fid'].to_numpy()[0])
# M_plot(U)
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
print('CNOT:')
print(np.linalg.det(CNOT))
print(local_invariants(CNOT))


print('iSWAP:')
# print(np.linalg.det(iSWAP))
# print(local_invariants(iSWAP))
'''Quantum Shannon Decomposition/ Cosine-Sine Decomposition'''
# (B1,B2), theta, (A1,A2) = la.cossin(U_swapped,p=4,q=4, separate=True)
# # B, cs, A = cossin(U,p=4,q=4)

# print('conditionals:')
# print([local_invariants(M) for M in [B1, B2, A1, A2]])
'''so far: the 4 controlled 2-qubit gates are roughly locally equivlant to iSWAP. TODO:How about the CCRy?'''
# # print(cs)
# # M_plot(MdgM(B))