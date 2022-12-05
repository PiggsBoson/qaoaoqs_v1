'''
Plot results
'''
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import re
import numpy as np


#Local scripts
from qaoaoqs.Dynamics import *
from qaoaoqs.quantum_manager import *
from result import *
from qaoaoqs.tools import *

'''
This is used to plot the maximum fidelity in each set of simulations against p,n and so on.
'''

def plot_data(data, args):
	'''Color schemes: https://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=6'''
	colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
	# colors = ['#7570b3','#1b9e77','#e7298a','#d95f02','#66a61e','#e6ab02']
	#Scheme 1: ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
	ftsz = 16 #Font size
	plt.rcParams.update({'font.size': ftsz})
	plt.rcParams['text.usetex'] = True

	# sns.set(style="darkgrid", font_scale=1.0)
	data.sort_values(by = [args.x], inplace=True)


	groupped = data.fillna(-1).groupby(args.group_by) #fillna() fills -1 to missing parameter entries due to version differences.
	i=0
	for name, group in groupped:
		if args.filter:
			if name != args.filter:
				continue
		if args.group_by_2nd:
			groupped_2nd = group.groupby(args.group_by_2nd)
			for name_2nd, group_2nd in groupped_2nd:
				line_color = colors[i]
				#Now only consiter group_by is approach
				if args.group_by == 'approach':
					line, = plt.plot(args.x, args.value,ls = '-', marker = 'o', data = group_2nd, color = line_color, alpha = 1.0, label = str(name).upper() + ' , ' + args.group_by_2nd + ' = ' +str(name_2nd))
				else: 
					line, = plt.plot(args.x, args.value,ls = '-', marker = 'o', data = group_2nd, color = line_color, alpha = 1.0, label = args.group_by + ' = ' +str(name) + ' , ' + args.group_by_2nd + ' = ' +str(name_2nd))
				print(line.get_data()) #Print highest fidelity
				if args.option == 'state_fid':
					plt.errorbar(args.x, 'state_fid_log' , yerr = 'state_fid_log_sd',ls = '-', marker = 'o', data = group_2nd, color = line_color, alpha = 0.5, capsize = 5, label='_nolegend_')
				i+=1
		else:
			line_color = colors[i]
			if args.x == "cp_str":
				line = plt.errorbar(args.x, args.value, yerr = 'log_fid_std', ls = '-', marker = 'o', data = group, color = line_color, alpha = 1.0,capsize = 5, label = name)
			else:
				line, = plt.plot(args.x, args.value,ls = '-', marker = 'o', data = group, color = line_color, alpha = 1.0, label = args.group_by + ' = ' +str(name))
				print(name)
				print(line.get_data()) 
				print(max(line.get_data()[1])) #Print highest fidelity

			if args.option == 'comparison':
				i+=1
				line.set_label('With control')
				plt.plot(args.x, 'log_fid_no',ls = '-', marker = 'o', data = group, color = colors[i], label = 'No Control') #only log fidelity is used

			elif args.option == 'state_fid':
				# line.set_label('Unitary')
				plt.errorbar(args.x, 'state_fid_log' , yerr = 'state_fid_log_sd',ls = '--', marker = 'o', data = group, color = line_color, alpha = 0.5, capsize = 5, label='_nolegend_')
			elif args.option == 'GRK':
				line.set_label('Unitary')
				plt.plot(args.x, 'GRK_fid_log',ls = '-', marker = 'o', data = group, label = 'GRK')
			elif args.group_by == 'approach':
				line.set_label(str(name).upper())
			i+=1

	#horizontal line for comparison. Not commonly used.
	# plt.plot(np.linspace(0,60), np.ones(50)*5.035443, ls = '-')#dp n=2,p=30: 5.867927
	# plt.plot(np.linspace(0,60), np.ones(50)*5.035443, ls = '-')#dp n=3,p=30: 3.972142
	# plt.plot(28.753447, 3.972142, marker = 'x', label = "Primitive PG")#Heisenberg n=3,p=30: 3.972142
	# plt.plot(41.900873, 5.035443, marker = 'x', label = "Primitive PG")#Heisenberg n=3,p=40: 5.035443
	# plt.plot(51.297791, 2.042751, marker = 'x', label = "Primitive PG")#Heisenberg n=4,p=50: 2.042751
	# plt.plot(30.285161, 6.059272, marker = 'x', label = "Primitive PG")	#Heisenberg n=2,p=30: 6.059272

	if args.x == "T_tot":
		plt.xlabel(r'$ T$', fontsize=ftsz)
	elif args.x == "TLS_T":
		plt.xlabel(r'$T$ (ns)', fontsize=ftsz)
	elif args.x == "cp_str":
		plt.xscale("log")
		plt.xlabel("Dipolar Coupling Strength", fontsize=ftsz)
	else:
		plt.xlabel(args.x, fontsize=ftsz)

	if args.value == "log_fid":
		plt.ylabel(r'$ -\log_{10}(1-F(\theta^*))$', fontsize=ftsz)
		plt.ylim(0, args.ylim)
	elif args.value == "fid":
		plt.ylabel(r'$ F(\theta_p)$', fontsize=ftsz)
	elif args.value == "T_tot":
		plt.ylabel(r'$ T$', fontsize=ftsz)
	elif args.value == "no_bath_fid":
		plt.ylabel(r'$F_{nom}$', fontsize=ftsz)
	elif args.value == "no_bath_fid_log":
		plt.ylabel(r'$-\log_{10}(1-F_{nom})$', fontsize=ftsz)
	elif args.value == "time_avg_intHam":
		plt.ylabel(r'$\langle G \rangle$', fontsize=ftsz)
	else:
		plt.ylabel(args.value, fontsize=ftsz)

	if args.title:
		plt.title(args.title)
	
	if args.legend:
		plt.legend(args.legend, fontsize = ftsz)
	elif args.legend_preset =='Lindn0':
		plt.legend([r'$T_1^{sys} = 500$ ns', 
		r'$T_1^{sys} = 1$ $\mu$s',
		r'$T_1^{sys} = 5$ $\mu$s'])
	elif args.legend_preset =='LindnX':
		plt.legend([r'$T_1^{sys} = 500$ ns, $T_1^{TLS} = 200$ ns', 
		r'$T_1^{sys} = 1$ $\mu$s, $T_1^{TLS} = 500$ ns',
		r'$T_1^{sys} = 5$ $\mu$s, $T_1^{TLS} = 1$ $\mu$s'])
		# plt.legend([r'$T_1^{sys} = 5$ $\mu$s, $T_1^{TLS} = 1$ $\mu$s'])
	elif args.legend_preset =='TLS2qb':
		plt.legend([r'$n_1=n_2=0$ (No bath)', '$n_1=1$,$n_2=0$', '$n_1=n_2=2$',])
	else:
		plt.legend(fontsize = ftsz)
	
	


	plt.xticks(fontsize=ftsz)	
	plt.yticks(fontsize=ftsz)	
	plt.tight_layout()
	plt.show()

def plot_ab(data):
	'''plotting the alphas and betas of a protocol.'''
	y = [-1 if i%2 else 1 for i in range(len(data['duration'].to_numpy()[0][0]))]
	y.insert(0,1)
	x = np.cumsum(data['duration'].to_numpy()[0][0])
	x = np.insert(x,0,0)
	print(x)
	plt.step(x,y)
	plt.show()

def plot_his(data, args):
	data.sort_values(by = [args.x], inplace=True)
	HIST_BINS = np.linspace(0, 5, 20)
	for i in range(len(data['duration'].to_numpy())):
		plt.cla()
		protocol = data['duration'].to_numpy()[i][0]
		grpA = [protocol[i] for i in range(len(protocol)) if not i%2]
		grpB = [protocol[i] for i in range(len(protocol)) if i%2]
		plt.hist([grpA, grpB],HIST_BINS)
		plt.savefig(str(i))

def plot_Bapat_n(data, args):
	'''Quantum approximate optimization of the long-range Ising model with a trapped-ion quantum simulator Fig. S1
	Not working for fixed p and T
	'''
	data.sort_values(by = [args.x], inplace=True)
	for i in range(len(data['duration'].to_numpy())):
		protocol = data['duration'].to_numpy()[i][0]
		grpA = [protocol[i] for i in range(len(protocol)) if not i%2]
		# grpB = [protocol[i] for i in range(len(protocol)) if i%2]
		plt.plot(grpA, label = i)
		# plt.plot(grpB)
	plt.legend()
	plt.show()

def plot_Bapat_p(data, args):
	'''Quantum approximate optimization of the long-range Ising model with a trapped-ion quantum simulator Fig. S1
	Normalized for different p
	'''
	data.sort_values(by = [args.x], inplace=True)
	for protocol in data['duration'].to_numpy():
		protocol = protocol[0]
		p = len(protocol)//2
		x = []
		for i in range(p):
			x.append(i/(p-1))
		grpA = [protocol[i]*p for i in range(len(protocol)) if not i%2]
		# grpB = [protocol[i] for i in range(len(protocol)) if i%2]
		plt.plot(x,grpA, label = p)
		# plt.plot(grpB)
	plt.legend()
	plt.show()

def ani_his(data,args):
	# HIST_BINS = np.linspace(0, 5, 20)

	# # histogram our data with numpy
	# n, _ = np.histogram(data['duration'].to_numpy()[0][0], HIST_BINS)

	# def prepare_animation(bar_container):

	# 	def animate(frame_number,data):
	# 		# simulate new data coming in
	# 		protocol = data[frame_number][0]
	# 		grpA = [protocol[i] for i in range(len(protocol)) if not i%2]
	# 		grpB = [protocol[i] for i in range(len(protocol)) if i%2]
	# 		protocol = [grpA, grpB]
	# 		n, _ = np.histogram(protocol, HIST_BINS)
	# 		for count, rect in zip(n, bar_container.patches):
	# 			rect.set_height(count)
	# 		return bar_container.patches
	# 	return animate

	# fig, ax = plt.subplots()
	# _, _, bar_container = ax.hist(data['duration'].to_numpy()[0][0], HIST_BINS)
	# ax.set_ylim(top=10)  # set safe limit to ensure that all data is visible.

	# ani = FuncAnimation(fig, prepare_animation(bar_container),frames = len(data['duration'].to_numpy()), fargs=(data['duration'].to_numpy(), ))
	# plt.show()
	'''https://stackoverflow.com/questions/35108639/matplotlib-animated-histogram'''
	from matplotlib.animation import FuncAnimation 
	data.sort_values(by = [args.x], inplace=True)
	HIST_BINS = np.linspace(0, 5, 20)
	def update_hist(num, data):
		plt.cla()
		protocol = data[num][0]
		grpA = [protocol[i] for i in range(len(protocol)) if not i%2]
		grpB = [protocol[i] for i in range(len(protocol)) if i%2]
		plt.hist([grpA, grpB], HIST_BINS)
		plt.ylim(0,15)
	fig, ax = plt.subplots()
	_, _, bar_container = ax.hist(data['duration'].to_numpy()[0][0], HIST_BINS)
	ax.set_ylim(top=10)
	
	ani = FuncAnimation(fig, update_hist,frames = len(data['duration'].to_numpy()), fargs=(data['duration'].to_numpy(), ) )
	plt.show()

def get_table(filepath) -> pd.DataFrame:

	iter_list = []
	duration_list = []
	fid_list = []

	with open(filepath, 'r') as f:
		a = re.findall(r"iter:(.*), protocol duration:([\s\S]*?)fid:(.*)",f.read())
		if a == []:
			pass
		else:
			iter_list.append( int(a[-1][0]) )
			duration_list.append( np.mat(a[-1][1]).tolist() )
			fid_list.append( float(a[-1][2]) )


	train_dict = {
		'iter': iter_list,
		'duration': duration_list,
		'fid': fid_list
	}


	train_df = pd.DataFrame(train_dict)

	return train_df



# def fidelity(data, params):
# 	'''
# 	Note: this only works for CS now!!!
# 	'''
# 	quma = sys_setup.setup(params)
# 	return quma.get_reward(data['duration'].to_numpy()[0][0])
	

def get_datasets(fpath, args = None) -> Result:
	datasets = []
	params = [] #Try declearation first to maintain good practice
	for root, dir, files in os.walk(fpath):
		if 'protocol.log' in files:
			param_path = open(os.path.join(root,'train_params.json'))
			params = json.load(param_path)
			# exp_name = 'batch_size = {}'.format( params['batch_size'])
			log_path = os.path.join(root,'protocol.log')
			experiment_data = get_table(log_path)
 
			for param in params:
				if param =='couplings':
					experiment_data.insert(len(experiment_data.columns), param, None)
					experiment_data.at[0,'couplings'] = params[param]
				else:
					experiment_data.insert(len(experiment_data.columns), param, params[param])
			datasets.append(experiment_data)
		else:
			print(fpath)
	#Only keep the highest fidelity one
	if isinstance(datasets, list):
		datasets = pd.concat(datasets, ignore_index=True)
	
	if args.x != "cp_str":	
		datasets = datasets[datasets["fid"]==datasets["fid"].max()]
	else:
		datasets["cp_str"] = [10**(-int(p)) for p in datasets["cp_str"]]

	if not 'T_tot' in datasets.columns:
		#Accomondate onlder versions
		datasets.insert(len(datasets.columns), 'T_tot', np.sum(datasets['duration'].to_numpy()[0]))
	
	#create a argparse.Namespace object of the parameters for simpler evaluation of the fidelity using existing packages
	from argparse import Namespace
	ns_params = Namespace(**params)
	
	result = Result(fpath, datasets, ns_params)
	if datasets["testcase"].values[0] in {"XmonTLS", 'TLSsec_bath_lowStr', 'Koch_1qb','TLSsec_bath', 'Lloyd_var1', 'TLSsec_bath_2qb'}:
		#Use physical timescale
		result.transmon_time(8)
	elif datasets["testcase"].values[0] in {'Koch_paper_1qb_noLind', 'Koch_paper_1qb'}:
		#Use physical timescale
		result.transmon_time(1)
	else:
		result.transmon_time(args.Ham_str)

	if args:
		if args.sanity_check:
			#For uneq case, the coupling constants are not kept track of, so can't do this check.   TODO: update the record-keeping later
			assert np.isclose(datasets["fid"].to_numpy()[0], result.compute_fid(), atol = 5e-3),\
				"The fidelity computed from the protocol does not agree with the record!"
		if args.value == "log_fid":    
			result.log_fid()
		elif args.value == "no_bath_fid" or args.value == 'no_bath_fid_log':    
			result.no_bath()
		elif args.value == 'HA_frac':
			result.HA_fraction()
		elif args.value == 'time_avg_intHam':
			result.time_avg_intHam()


		if args.option == 'comparison':
			#the fidelity without control for comparision
			result.no_opt()
			# datasets.insert(len(datasets.columns), 'no_opt', no_opt(datasets, params))
		elif args.option == 'state_fid':
			result.state_fid(args.temp, args.state_samples)
		elif args.option == 'GRK':
			result.GRK_fidelity(args.temp)
		elif args.option == 'call':
			method = getattr(result,args.func_n)
			print(method())
		elif  args.option == 'display_data':
			print(result.data.to_string())
		
		if args.x == 'cp_str':
			result.fid_stat()

	return result



def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('logdir', nargs='*')
	parser.add_argument('--legend', nargs='*', default= None)
	parser.add_argument('--legend_preset', default= None)
	parser.add_argument('--x', default='T_tot')
	parser.add_argument('--value', 
						choices=['fid','log_fid','T_tot','iter', 'no_bath_fid','no_bath_fid_log', 'HA_frac', 'time_avg_intHam'],
						default='log_fid')
	parser.add_argument('--group_by',default='p')
	parser.add_argument('--group_by_2nd',default=None)
	parser.add_argument('--filter', choices=['p','n','lind_gamma'],default=None)
	# parser.add_argument('--comparision', type=bool, default=False)
	parser.add_argument('--title',type=str, default=None)
	parser.add_argument('--option',choices=['plot', 'state_fid', 'comparison', 'plot_ab','plot_his','ani_his', 'Bapat_n','Bapat_p', 'GRK', 'grad', 'call','display_data'],default='plot')
	parser.add_argument('--func_n',help='function name to call in the result class')
	parser.add_argument('--temp',choices=['zero', 'inf', 'rand'],default='zero', help='temperature')
	parser.add_argument('--state_samples', type = int, default=100)
	parser.add_argument('--sanity_check',type=bool, default=False)
	parser.add_argument('--ylim',type = int, default=11)
	parser.add_argument('--Ham_str', type = int, default = 1)

	args = parser.parse_args()

	data = pd.DataFrame()
	
	for logdir in args.logdir:
		result = get_datasets(logdir, args)
		# results.append(result)
		data=data.append(result.data, ignore_index=True)
	
	data.rename(columns={"env_dim": "n"}, inplace = True) #For better readability
	data[['n']] = data[['n']].astype(int)
	
	if args.option == 'plot_ab':
		assert len(data) == 1, "Should only enter one experiment!"
		plot_ab(data)
	elif args.option == 'plot_his':
		plot_his(data, args)
	elif args.option == 'ani_his':
		ani_his(data, args)
	elif args.option == 'Bapat_n':
		plot_Bapat_n(data, args)
	elif args.option == 'Bapat_p':
		plot_Bapat_p(data, args)
	elif args.option == 'grad':
		ev1, _ = np.linalg.eig([[1,1],[1,1]])
		grad,heis = result.compute_grad()
		print("The gradient is :", grad)
		ev, _ = np.linalg.eig(heis)
		print("The eigenvalues of the Hessian is", ev)
	elif args.option == 'plot':
		plot_data(data, args)


if __name__ == "__main__":
	main()
