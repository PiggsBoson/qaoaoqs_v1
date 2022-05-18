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
    v_0 = np.array([0,0,1], dtype = float)
    v_0 /= np.linalg.norm(v_0)
    v_pm0 = np.array([v_0[0]+1.0j*v_0[1], v_0[0]-1.0j*v_0[1]])/2
    alpha = 1/200
    t_array = np.linspace(0,2/alpha,50)
    observable = pop_0
    N_array = [1,3,5,10,20,50] #range(1,7)##number of bath spins
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
    plt.rcParams.update({'font.size': 14})

    t_TLS = t_array/(8 *2*np.pi)
    y_infarray = []
    i = 0
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
        plt.plot(t_TLS, y_array ,color = colors[i], label = 'n='+str(N))

        #comparing with numerical simulation to check correctness
        # H = get_Ham(N, True, 'Breuer_dp', alpha = alpha_scaled)
        # sys_init = v2rho(v_0[2], v_pm0)
        # bath_init = np.identity(2**N, dtype='complex128') #unpolarized state of bath
        # bath_init /= 2**N #Normalize
        # rho0 = np.kron(sys_init,bath_init)
        # result_num =[]
        # for t in t_array:
        #     result_num.append(evolveNmeasure(rho0, H, t, observable))
        # plt.plot(t_array, result_num, '-x')
        i+=1
    for t in t_array:
        f12inf, f3inf = f_123inf(t,alpha)
        v_pmtinf = f12inf*v_pm0
        v3tinf = f3inf*v_0[2]
        rho_inf = v2rho(v3tinf, v_pmtinf)
        y_infarray.append(np.trace(rho_inf@observable))
    plt.plot(t_TLS, y_infarray, color = colors[i],label = 'n=inf')

    plt.xlabel(r'$t$ (ns)')
    plt.ylabel(r'$\rho_{00}$')
    # plt.ylabel('Re(rho_01)')
    plt.legend()
    plt.show()

main()