'''
Non-Markovian dynamics in a spin star system: Exact solution and approximation techniques
'''
from cmath import cos
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from decoherence import *

def eig_dim(N, j):
    result = special.comb(N, (N/2-j)) - special.comb(N, (N/2-j-1))
    return result

def h(j,m):
    return np.sqrt((j+m)*(j-m+1))

def f_123(t, N, alpha):
    result_12 = 0
    result_3 = 0
    parity = N%2/2
    alphat = alpha*t
    for i in range(N//2+1):
        j = i+parity
        n_Nj = eig_dim(N,j)
        for k in range(i+1):           
            m = k+parity
            result_12 += n_Nj*cos(2*h(j,m)*alphat)*cos(2*h(j,-1*m)*alphat)
            result_3 +=n_Nj*cos(4*h(j,m)*alphat)
            if m:
                m *= -1
                result_12 += n_Nj*cos(2*h(j,m)*alphat)*cos(2*h(j,-1*m)*alphat)
                result_3 += n_Nj*cos(4*h(j,m)*alphat)
    return result_12/2**N, result_3/2**N


def g(t, alpha):
    alphat = alpha*t
    return -1*alphat*np.exp(-2*alphat**2)*np.sqrt(np.pi/2)*special.erfi(np.sqrt(2)*alphat)

def f_123inf(t,alpha):
    g_val = g(t, alpha)
    return 1+g_val, 1+2*g_val



def main():
    v_0 = np.array([1,0,0], dtype = float)
    v_0 /= np.linalg.norm(v_0)
    v_pm0 = np.array([v_0[0]+1.0j*v_0[1], v_0[0]-1.0j*v_0[1]])/2
    alpha = 1/200
    t_array = np.linspace(0,3/alpha,50)
    observable = coh_01
    N_array = range(1,7)#[1,3,5,10,20,50] #number of bath spins

    y_infarray = []
    for N in N_array:
        alpha_scaled = alpha/np.sqrt(N) #Use scaled version for exact dynamics with N qubits
        y_array = []
        for t in t_array:
            f12, f3 = f_123(t, N, alpha_scaled)
            v_pmt = f12*v_pm0
            v3t = f3*v_0[2]
            rho_N = v2rho(v3t,v_pmt)
            temp = np.trace(rho_N@observable)
            y_array.append(temp)
        plt.plot(t_array, y_array,'-o',label = 'n='+str(N))

        #comparing with numerical simulation to check correctness
        H = get_Ham(N, True, 'Breuer_dp', alpha = alpha_scaled)
        sys_init = v2rho(v_0[2], v_pm0)
        bath_init = np.identity(2**N, dtype='complex128') #unpolarized state of bath
        bath_init /= 2**N #Normalize
        rho0 = np.kron(sys_init,bath_init)
        result_num =[]
        for t in t_array:
            result_num.append(evolveNmeasure(rho0, H, t, observable))
        plt.plot(t_array, result_num, '-x')

    # for t in t_array:
    #     f12inf, f3inf = f_123inf(t,alpha)
    #     v_pmtinf = f12inf*v_pm0
    #     v3tinf = f3inf*v_0[2]
    #     rho_inf = v2rho(v3tinf, v_pmtinf)
    #     y_infarray.append(np.trace(rho_inf@observable))
    # plt.plot(t_array, y_infarray, label = 'n=inf')

    plt.xlabel('t')
    plt.ylabel('Re(rho_01)')
    plt.legend()
    plt.show()

main()