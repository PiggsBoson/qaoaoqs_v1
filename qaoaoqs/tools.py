'''Tool functions and important matrices'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

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

CPHASE = np.array([[1, 0, 0, 0], 
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]])

H = np.array([[1, 0], 
            [0, 1]])

#Pauli matrices
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
sigma_y = 1.0j * np.array([[0.0, -1.0], [1.0, 0.0]])
Id = np.identity(2)

'''Functions'''
def U_Sys(U_full, nb, part):
    '''Returns a submatrix on the system, given bath index
    https://stackoverflow.com/questions/19161512/numpy-extract-submatrix'''
    return U_full[part[0]::nb, part[1]::nb]

def MdgM(M):
    return M.conjugate().transpose() @ M

def M_plot(M):
    '''plotting a matrix with real and imaginary parts seperated'''
    f, (ax1, ax2) = plt.subplots(1, 2)
    real = ax1.matshow(np.real(M))
    f.colorbar(real, ax=ax1)
    imag = ax2.matshow(np.imag(M))
    f.colorbar(imag, ax=ax2)
    plt.show()

def ptrace(M, ns, nb, out='b'):
    '''partial trace
    https://www.peijun.me/reduced-density-matrix-and-partial-trace.html'''
    M_tensor = M.reshape([ns, nb, ns, nb])
    if out =='b':
        result = np.trace(M_tensor, axis1=1, axis2=3)
    elif out == 's':
        result = np.trace(M_tensor, axis1=0, axis2=2)
    return result

def local_invariants(Uni):
    Q = np.array([[1, 0, 0, 1.0j],
                [0, 1.0j, 1, 0],
                [0, 1.0j, -1, 0],
                [1, 0, 0, -1.0j]])
    Q *= 1/np.sqrt(2)
    Qdg = Q.conjugate().T
    m = (Qdg @ Uni @Q).T @ Qdg@ Uni@Q
    Udet = np.linalg.det(Uni) 
    G1 = np.trace(m)**2/(16*Udet)
    G2 = (np.trace(m)**2 - np.trace(m@m))/(4*Udet)
    values = [G1.real, G1.imag, G2.real, G2.imag]
    for i, x in enumerate(values):
        if np.abs(x) <1e-12:
            values[i] = 0
    G1 = complex(values[0], values[1])
    G2 = complex(values[2], values[3])
    return G1, G2

def local_invariants_angles(Uni):
    Q = np.array([[1, 0, 0, 1.0j],
                [0, 1.0j, 1, 0],
                [0, 1.0j, -1, 0],
                [1, 0, 0, -1.0j]])
    Q *= 1/np.sqrt(2)
    Qdg = Q.conjugate().T
    m = (Qdg @ Uni @Q).T @ Qdg@ Uni@Q
    d,v =  np.linalg.eig(m)
    theta = np.arccos(np.real(d))
    # theta_s = np.arcsin(np.imag(d))
    # print(theta - theta_s) #TODO: check why they are not equal
    coeff = np.array([[1, -1, 1],
                    [1, 1, -1],
                    [-1, -1, -1]])
                    # [-1, 1, 1]])
    c = np.linalg.solve(coeff, theta[:-1])
    return c/np.pi
    
def fidelity(W, U, N_s, N_b):
    if N_b == 0:
        result = np.absolute(np.trace(W.conjugate().transpose() @ U) / N_s) ** 2
    else:
        W_f = np.kron(W, np.identity(N_b))
        Q = W_f.conjugate().transpose() @ U
        Q_b = ptrace(Q, N_s, N_b, out='s')
        result = np.absolute(np.trace(la.sqrtm(Q_b.conjugate().transpose() @ Q_b)) / (N_s*N_b)) ** 2
    return result

def plot_bloch_sphere(density_mtx):
    """Copied from https://pennylane.ai/qml/demos/tutorial_haar_measure.html
    """
    bloch_vectors = np.array([[np.trace(np.dot(rho, sigma_x)).real,
                    np.trace(np.dot(rho, sigma_y)).real,
                    np.trace(np.dot(rho, sigma_z)).real] for rho in density_mtx])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(30, 45)
    ax.dist = 7

    # Draw the axes (source: https://github.com/matplotlib/matplotlib/issues/13575)
    x, y, z = np.array([[-1.5,0,0], [0,-1.5,0], [0,0,-1.5]])
    u, v, w = np.array([[3,0,0], [0,3,0], [0,0,3]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.05, color="black", linewidth=0.5)

    ax.text(0, 0, 1.7, r"|0⟩", color="black", fontsize=16)
    ax.text(0, 0, -1.9, r"|1⟩", color="black", fontsize=16)
    ax.text(1.9, 0, 0, r"|+⟩", color="black", fontsize=16)
    ax.text(-1.7, 0, 0, r"|–⟩", color="black", fontsize=16)
    ax.text(0, 1.7, 0, r"|i+⟩", color="black", fontsize=16)
    ax.text(0,-1.9, 0, r"|i-⟩", color="black", fontsize=16)

    ax.scatter(
        bloch_vectors[:,0], bloch_vectors[:,1], bloch_vectors[:, 2], c='#e29d9e', alpha=0.3
    )
    plt.show()