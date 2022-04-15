###########################################################################
#                            example 17                                   #
#  In this script we demonstrate how to apply the matvec function         #
#  to define the Lundblad equation for a two-leel system, and solve it    #
#  using the evolve funcion.                                              #
###########################################################################
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d # Hilbert space spin basis_1d
from quspin.tools.evolution import evolve
from quspin.tools.misc import get_matvec_function
import numpy as np
from six import iteritems # loop over elements of dictionary
import matplotlib.pyplot as plt # plotting library
#
###### model parameters
#
L=3 # one qubit
delta=1.0 # detuning
Omega_0=np.sqrt(2) # bare Rabi frequency
Omega_Rabi=np.sqrt(Omega_0**2+delta**2) # Rabi frequency
gamma = 0.5*np.sqrt(3) # decay rate
#
##### create Hamiltonian to evolve unitarily
# basis
basis=spin_basis_1d(L,pauli=-1) # uses convention "+" = [[0,1],[0,0]]
# site-coupling lists
hx_list=[[Omega_0,i] for i in range(L)]
hz_list=[[delta,i] for i in range(L)]
# static opstr list 
static_H=[['x',hx_list],['z',hz_list]]
# hamiltonian
H=hamiltonian(static_H,[],basis=basis,dtype=np.float64)
print('Hamiltonian:\n',H.toarray())
#
##### create Lindbladian
# site-coupling lists
L_list=[[1.0j,i] for i in range(L)]
# static opstr list 
static_L=[['+',L_list]]
# Lindblad operator
L=hamiltonian(static_L,[],basis=basis,dtype=np.complex128,check_herm=False)
print('Lindblad operator:\n',L.toarray())
# pre-compute operators for efficiency
L_dagger=L.getH()
L_daggerL=L_dagger*L
#
#### determine the corresponding matvec routines ####
#
matvec=get_matvec_function(H.static)
#
#
##### define Lindblad equation in diagonal form
#
# fast function using matvec (not as memory efficient)
#
def Lindblad_EOM_v3(time,rho,rho_out,rho_aux):
	"""
	This function solves the complex-valued time-dependent GPE:
	$$ \dot\rho(t) = -i[H,\rho(t)] + 2\gamma\left( L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho \} \right) $$
	"""
	rho = rho.reshape((H.Ns,H.Ns)) # reshape vector from ODE solver input
	### Hamiltonian part
	# commutator term (unitary
	# rho_out = H.static.dot(rho))
	matvec(H.static  ,rho  ,out=rho_out  ,a=+1.0,overwrite_out=True)
	# rho_out -= (H.static.T.dot(rho.T)).T // RHS~rho.dot(H) 
	matvec(H.static.T,rho.T,out=rho_out.T,a=-1.0,overwrite_out=False)
	# 
	for func,Hd in iteritems(H._dynamic):
		ft = func(time)
		# rho_out += ft*Hd.dot(rho)
		matvec(Hd  ,rho  ,out=rho_out  ,a=+ft,overwrite_out=False)
		# rho_out -= ft*(Hd.T.dot(rho.T)).T 
		matvec(Hd.T,rho.T,out=rho_out.T,a=-ft,overwrite_out=False)
	# multiply by -i
	rho_out *= -1.0j
	#
	### Lindbladian part (static only)
	# 1st Lindblad term (nonunitary)
	# rho_aux = 2\gamma*L.dot(rho)
	matvec(L.static  ,rho             ,out=rho_aux  ,a=+2.0*gamma,overwrite_out=True)
	# rho_out += (L.static.T.conj().dot(rho_aux.T)).T // RHS~rho_aux.dot(L_dagger) 
	matvec(L.static.T.conj(),rho_aux.T,out=rho_out.T,a=+1.0,overwrite_out=False) 
	# anticommutator (2nd Lindblad) term (nonunitary)
	# rho_out += gamma*L_daggerL.static.dot(rho)
	matvec(L_daggerL.static  ,rho  ,out=rho_out  ,a=-gamma,overwrite_out=False)
	# # rho_out += gamma*(L_daggerL.static.T.dot(rho.T)).T // RHS~rho.dot(L_daggerL) 
	matvec(L_daggerL.static.T,rho.T,out=rho_out.T,a=-gamma,overwrite_out=False) 
	
	return rho_out.ravel() # ODE solver accepts vectors only
#
# define auxiliary arguments
EOM_args=(np.zeros((H.Ns,H.Ns),dtype=np.complex128,order="C"),    # auxiliary variable rho_out
		  np.zeros((H.Ns,H.Ns),dtype=np.complex128,order="C"),  ) # auxiliary variable rho_aux
#
##### time-evolve state according to Lindlad equation
# define real time vector
t_max=6.0
time=np.linspace(0.0,t_max,101)
# define initial state
rho0=np.array([[0.5,0.5j],[-0.5j,0.5]],dtype=np.complex128)
# fast solution (but 3 times as memory intensive), uses Lindblad_EOM_v3
rho_t = evolve(rho0,time[0],time,Lindblad_EOM_v3,f_params=EOM_args,iterate=True,atol=1E-12,rtol=1E-12) 
#
# compute state evolution
population_down=np.zeros(time.shape,dtype=np.float64)
for i,rho_flattened in enumerate(rho_t):
	rho=rho_flattened.reshape(H.Ns,H.Ns)
	population_down[i]=rho_flattened[1,1].real
#
##### plot population dynamics of down state
#
plt.plot(Omega_Rabi*time, population_down)
plt.xlabel('$\\Omega_R t$')
plt.ylabel('$\\rho_{\\downarrow\\downarrow}$')
plt.xlim(time[0],time[-1])
plt.ylim(0.0,1.0)
plt.grid()
plt.tight_layout()
plt.show()