'''
Correlated projection operator approach to non-Markovian dynamics in spin baths
'''
from cmath import cos, sin
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from decoherence import *

def eig_dim(N, j):
    '''
    the multiplicity of each (j,m) subspace with j.
    '''
    result = special.comb(N, (N/2-j)) - special.comb(N, (N/2-j-1))
    return result

def Omega(m, omega0, A, fac):
    return omega0*fac+4*A*(fac*m+1/2)

def mu(j,m,omega0, A, fac):
    return np.sqrt(Omega(m,omega0, A, fac)**2/4 + 4*A**2 * (j*(j+1) - m*(m+fac) ))

def rho_t(t, N, omega0, A, rho0):
    parity = N%2/2
    gt = 0
    rhopm = 0
    # phase = np.exp(1.0j*omega0*t)
    for i in range(N//2+1):
        j = i+parity
        temp_p = 0
        temp_pm = 0
        for k in range(i+1):
            m = k+parity
            Omega_mp = Omega(m, omega0, A, 1)
            mu_jmp = mu(j,m,omega0, A, 1)
            Omega_mm = Omega(m, omega0, A, -1)
            mu_jmm = mu(j,m,omega0, A, -1)
            temp_p += np.square(cos(mu_jmp*t)) + np.square(Omega_mp)/(4*np.square(mu_jmp))* np.square(sin(mu_jmp*t))
            temp_pm += (cos(mu_jmp*t) - 1.0j*Omega_mp/(2*mu_jmp)*sin(mu_jmp*t)) * (cos(mu_jmm*t) + 1.0j*Omega_mm/(2*mu_jmm)*sin(mu_jmm*t))
            if m:
                m *= -1
                Omega_mp = Omega(m, omega0, A, 1)
                mu_jmp = mu(j,m,omega0, A, 1)
                Omega_mm = Omega(m, omega0, A, -1)
                mu_jmm = mu(j,m,omega0, A, -1)
                temp_p += np.square(cos(mu_jmp*t)) + np.square(Omega_mp)/(4*np.square(mu_jmp))* np.square(sin(mu_jmp*t))
                temp_pm += (cos(mu_jmp*t) - 1.0j*Omega_mp/(2*mu_jmp)*sin(mu_jmp*t)) * (cos(mu_jmm*t) + 1.0j*Omega_mm/(2*mu_jmm)*sin(mu_jmm*t))
        temp_p *= eig_dim(N,j)
        gt += temp_p
        temp_pm *= eig_dim(N,j)
        rhopm+= temp_pm
    gt /= 2**N
    rhopm /= 2**N #there was a phase here but the results agree without the phase, see the eq between Eq. 30 and 31. Should be the same amplitude, but we are just plotting the real part.
    v0 = 2*rho0[0,0]-1 #The initial Bloch vector
    Pp = ((2*gt-1)*v0 +1)/2 #Their solution is based on bloch vector and only applicable for certain initial state. This conversion makes it general.
    rhopm *= rho0[0,1]
    return np.array([[Pp, rhopm],[np.conj(rhopm),1-Pp]])


def main():
    A = 1
    N_array = [1,2,3,4,5,6,10,20,50,100]#range(1,7)#
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
    t_array = np.linspace(0,50/A,100)
    plot_opt = pop_0
    omega0 = 1
    sys_init = np.array([1.,0.])#np.random.uniform(size =2)
    sys_init /= np.linalg.norm(sys_init)
    rho0 = np.outer(sys_init,sys_init.conj().T)
    plt.rcParams.update({'font.size': 14})

    fig, ax1 = plt.subplots()
    l, b, h, w = .45, .2, .3, .4
    ax2 = fig.add_axes([l, b, w, h])
    i= 0
    for N in N_array:
        # A = A/np.sqrt(N)
        y_array = []
        for t in t_array:
            y_array.append(np.trace(rho_t(t, N, omega0, A, rho0)@plot_opt))
        ax1.plot(t_array, y_array,color = colors[i],label = 'n='+str(N))
        ax2.plot(t_array[:30], y_array[:30],color = colors[i],label = 'n='+str(N))
        # plt.plot(t_array, y_array,'-o',label = 'n='+str(N))

        #comparing with numerical simulation to check correctness
        # H = get_Ham(N, True, 'Breuer_iso', A = A)
        # bath_init = np.identity(2**N, dtype='complex128') #unpolarized state of bath
        # bath_init /= 2**N #Normalize
        # rho0_sb = np.kron(rho0,bath_init)
        # result_num =[]
        # for t in t_array:
        #     result_num.append(evolveNmeasure(rho0_sb, H, t, plot_opt))
        # plt.plot(t_array, result_num, '-x')
        i+=1
    ax1.set_ylim(0,1)
    ax1.set_xlabel(r't')
    # plt.ylabel('Re(rho_01)')
    ax1.set_ylabel(r'$\rho_{00}$')
    ax1.legend() 

    ax2.set_xticks([0.00, 0.01,0.02, 0.03])
    # ax2.set_ylim(0.8,1)
    ax2.set_yticks([0.8, 0.9,1.0])
    ax2.tick_params(labelsize= 8)
    plt.show()

main()