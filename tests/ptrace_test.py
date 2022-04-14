#Passed!
import numpy as np

sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
sigma_y = 1.0j * np.array([[0.0, -1.0], [1.0, 0.0]])

def ptrace(rho, remain, dim_a, dim_b):
    #The following is partial trace according to Peijun Zhu https://www.peijun.me/reduced-density-matrix-and-partial-trace.html
    rho_tensor = rho.reshape([dim_a, dim_b, dim_a, dim_b])
    if remain == 'a':
        return np.trace(rho_tensor, axis1=1, axis2=3)
    elif remain == 'b':
        return np.trace(rho_tensor, axis1=0, axis2=2)

# #Testcase 1
# psi_a1 = np.array([np.sqrt(0.3), np.sqrt(0.7)])
# psi_b1 = np.array([0.5, 0.5,0.5,0.5])
# rho_a1 = np.outer(psi_a1, psi_a1.conj().T)
# rho_b1 = np.outer(psi_b1, psi_b1.conj().T)
# rho1 = np.kron(rho_a1, rho_b1)
# print(rho_a1)
# print(rho_b1)
# print(rho_a1 -  ptrace(rho1, 'a', 2,4))
# print(rho_b1 -  ptrace(rho1, 'b', 2,4))

#Testcase 2
psi_a2 = np.random.rand(2**2)
psi_a2 /= np.sqrt(psi_a2.conj().T @ psi_a2)
psi_b2 = np.random.rand(2**3)
psi_b2 /= np.sqrt(psi_b2.conj().T @ psi_b2)
rho_a2 = np.outer(psi_a2, psi_a2.conj().T)
rho_b2 = np.outer(psi_b2, psi_b2.conj().T)
rho2 = np.kron(rho_a2, rho_b2)
print(rho_b2)
print(rho_b2)
print([0 for v in rho_a2 -  ptrace(rho2, 'a', 4,8) for x in v if x<1e-5 ])
print(rho_b2 -  ptrace(rho2, 'b', 4,8))