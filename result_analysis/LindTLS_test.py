'''
Check if the weak parameter test case is implemented correctly
'''
import analysis
from qaoaoqs.tools import *
from qaoaoqs.sys_setup import *

A = np.random.uniform(0.5, 5, 2)
A /= 1000 #Scale to desired strength
result_temp = analysis.get_datasets('LindTLS/1qubit/T_PGSecB_n2eq_T5uT1u_p30T40')
quma_strong = setup(result_temp.params, couplings = A*8)
quma_weak = setup(result_temp.params, couplings = A, alt_testcase='TLSsec_bath_lowStr')
quma_weak.T_tot = quma_strong.T_tot*8
protocol = np.array(result_temp.data['duration'].to_numpy()[0][0])
print(quma_strong.get_reward(protocol))
print(quma_weak.get_reward(protocol))
