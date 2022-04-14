import numpy as np
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/william/OneDrive - HKUST Connect/Codes/QAOA/PGQAOA_OpenQuantumSystem/')
from Dynamics import *
import matplotlib.pyplot as plt

def plotmtx(M):
    plt.matshow(np.real(M))
    plt.colorbar()
    plt.matshow(np.imag(M))
    plt.colorbar()
    plt.show()

X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.array([[1.0, 0.0], [0.0, -1.0]])
Y = 1.0j * np.array([[0.0, -1.0], [1.0, 0.0]])

XX = np.kron(X,X)
YY = np.kron(Y,Y)
g=0.05
T= t = np.pi / (2*g)
H1 = g/2*(XX+YY)
u0 = np.eye(4)

simulator1 = Dyna(u0, H1, N=4)
u0T = simulator1.simulate_closed(T)


H2 = g/2*XX
H3 = g/2*YY
simulator2 = Dyna(u0, H2, N=4)
u1= simulator2.simulate_closed(T)
simulator3 = Dyna(u1, H3, N=4)
u3T = simulator3.simulate_closed(T)
plotmtx(u0T-u3T)