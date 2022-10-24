#Modified by Zhibo for central spin model
#Feb. 05 2020
import numpy as np
import scipy.linalg as la
import scipy as sp
import scipy.optimize as opt
import tensorflow as tf

from qaoaoqs.reinforce import Reinforce
from qaoaoqs.quantum_manager import *
import qaoaoqs.scp_convex_mod as scp
import cvxpy as cp

import os
import time
from tqdm import trange
import argparse
import datetime
import dateutil.tz
import glob
import json
import pickle
import logging

import nevergrad as ng

import psutil
from multiprocessing import Process
from multiprocessing import Pool
from joblib import Parallel, delayed

import qaoaoqs.sys_setup as sys_setup

def setup_logger(name, log_file, level=logging.INFO):
	"""Function setup as many loggers as you want"""
	formatter = logging.Formatter('%(asctime)s %(message)s')

	handler = logging.FileHandler(log_file)
	handler.setFormatter(formatter)

	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)

	return logger


def parse_args():
	desc = "TensorFlow implementation of 'Policy Gradient based Quantum Approximate Optimization Algorithm'"
	parser = argparse.ArgumentParser(description=desc)
	arg_lists = []

	def add_argument_group(name):
		arg = parser.add_argument_group(name)
		arg_lists.append(arg)
		return arg
	
	# system parameters
	sys_arg = add_argument_group('System')
	sys_arg.add_argument('--path', help='path for result')
	sys_arg.add_argument('--exp_name', help='Name of the simulation')

	# qunatum control
	qc_arg = add_argument_group('QuantumControl')
	qc_arg.add_argument('--p', type=int,
						default=10, help='Numbers of bangs')
	qc_arg.add_argument('--approach', choices=['qaoa', 'blackbox', 'pg', 'es',
											   'brs', 'ars', 'scp'], default='pg', help='different approachs to the problem')
	qc_arg.add_argument('--qaoa_method', choices=['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr',
												  'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'], default='trust-constr', help='optimization method for the scipy optimize')
	#Physical parameters
	qc_arg.add_argument('--testcase', choices=['lind', 'cs_au', 'ns_cs', 'ns_lind', 'Heis_lab',
						'dipole', 'XmonTLS', 'Xmon_nb','NoBath','dipole_VarStr', 
						'2qbiSWAP', '2qbCPHASE', 'XXnYY', 'XXpm', 'XXYY_X', 
						'Lloyd_2qb', 'Lloyd_3qb', 'Lloyd_var1',
						'TLS_bb', 'Heis_bbzz', 'TLSsec_bath', 'TLSsec_bath_lowStr', 'Koch_1qb',
						'Koch_paper_1qb_noLind', 'Koch_paper_1qb', 'TLSsec_bath_2qb',
						'1qb1anci'],
						default='cs_au', help='different test case for the problem')
	qc_arg.add_argument('--env_dim', type=int, default=0,
						help='number of bath spins in central spin model')
	qc_arg.add_argument('--env_dim2', type=int, default=0,
						help='number of bath spins coupled to the second qubit')
	qc_arg.add_argument('--fid_fix', choices=['abs', 'barrier'],
					   default=None, help='method to fix time negativity')
	qc_arg.add_argument('--cs_coup', choices=['eq', 'uneq'],
						default='eq', help='whether the coupling constants are equal or not')
	qc_arg.add_argument('--cs_coup_type', choices=['Heis', 'dipole'],
						default='dipole', help='The physical model for the coupling')
	qc_arg.add_argument('--cs_coup_scale', type=float, default= 1.0,
						help='scaling of the coupling strength')
	qc_arg.add_argument('--au_uni', choices=['rand', 'Had', 'ST', 'Zrot','T', 'S','CNOT','iSWAP'],
						default='rand', help='The target unitary')
	qc_arg.add_argument('--target_angle',
						default=1/8, help='The target angle of rotation as the multiple of pi')
	qc_arg.add_argument('--measure', choices=['quantum', 'noise', 'noise_gate', 'plain', 'noise_avg',
											  'noise_gate_avg'], default='plain', help='different settings for fidelity measurement')
	qc_arg.add_argument('--uneq', choices=['normal', 'uniform'],
						default = 'normal', help='coupling constants if unequal coupling')
	qc_arg.add_argument('--cp_str', type=float,
						default = 1.0, help='coupling strength')
	qc_arg.add_argument('--b_temp', choices=['zero', 'inf'],
						default = 'zero', help='Bath temperature')					
	#Parameters for dynamics
	qc_arg.add_argument('--impl', choices=['qutip', 'numpy', 'quspin', 'vec'],
					default = 'numpy', help='different implementations of the dynamics')
	qc_arg.add_argument('--ode_steps', type=int,
						default=10, help='Numbers of steps for each time interval for the ode solver')	
	#Lindblad parameters
	qc_arg.add_argument('--deco_type', default='-', help='The Lindblad operator')
	qc_arg.add_argument('--T1_TLS', type=float, default= 1.0,
						help='The T1 time of all the TLS')
	qc_arg.add_argument('--T1_sys', type=float, default= 1.0,
						help='The T1 time of the system qubits')
	
	#Protocol-related parameters
	qc_arg.add_argument('--fid_adj', choices=['t', 'others'],
					   default=None, help='adjustments to fidelity for certain purposes')
	qc_arg.add_argument('--T_tot', type=float,
						default=3.0, help='Total duration of the protocol')
	qc_arg.add_argument('--fid_target', type=float,
						default=1-1e-9, help='Target fidelity')
	qc_arg.add_argument('--protocol_renormal', type=bool,
						default=False, help='Whether renormalizing the protocol in computing the fidelity')

	

	# network
	net_arg = add_argument_group('Network')
	net_arg.add_argument('--softplus', action='store_true',
						 help='whether or not to use the lstm for the network')

	learn_arg = add_argument_group('Learning')
	learn_arg.add_argument('--batch_size', '-b', type=int, default=128)
	learn_arg.add_argument('--num_iters', type=int, default=1000)
	learn_arg.add_argument('--learning_rate', '-lr', type=float, default=1e-2)
	learn_arg.add_argument('--decay_steps', type=int, default=50)
	learn_arg.add_argument('--n_experiments', '-e', type=int, default=1)
	learn_arg.add_argument('--seed', '-s', type=int, default=11,
						   help='Randon seed in order to reproduce the results')
	learn_arg.add_argument('--loadmodel_dir', type=str,
						   help='The pretrained model')
	learn_arg.add_argument('--lr_decay', action='store_true',
						   help='Add learning rate decay')
	learn_arg.add_argument('--optimizer', choices=['adam', 'sgd', 'rms'],
						   default='adam', help='different optimizer to choose from')
	learn_arg.add_argument('--distribution', choices=['normal', 'beta', 'logit-normal'],
						   default='normal', help='distribution for sampling durations') #TODO: implement beta
	learn_arg.add_argument('--scale', type=float, default=1.0,
							help = 'sacing factor for bounded distributions')



	args = parser.parse_args()

	return args


def train(seed, exp_dir):
	global args

	# making the logging file
	assert not os.path.exists(exp_dir),\
		'Experiment directory {0} already exists. Either delete the directory, or run the experiment with a different name'.format(
			exp_dir)
	os.makedirs(exp_dir, exist_ok=True)
	

	quma = sys_setup.setup(args) #Set up physical parameters of the system

	###############################################

	# Set random seeds
	tf.set_random_seed(seed)
	np.random.seed(seed)
	params = vars(args)
	params['seed'] = seed
	params['couplings'] = quma.couplings.tolist()
	###############################################


	logger = setup_logger('train', os.path.join(exp_dir, "train.log"))
	logger_protocol = setup_logger(
		'protocol', os.path.join(exp_dir, "protocol.log"))

	import json
	with open(os.path.join(exp_dir, "train_params.json"), "w") as f:
		# change the namespace to dict
		# https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
		json.dump(params, f)

	num_cpus = psutil.cpu_count(logical=True)

	if args.approach == 'qaoa':
		x0 = np.random.normal(loc=0.5, scale=0.1, size=args.p)

		if args.qaoa_method == 'trust-constr':
			def mycallback(x, state):
				score = quma.get_reward(x)
				logger.info("iter: {}, loss: {}, mean_reward: {}, max_reward: {},  test_reward: {}, his_reward: {}, entropy: {}".
							format(state.niter, -score, score, score, score, score, 0.0))
		else:
			iter_i = 0

			def mycallback(x):
				nonlocal iter_i
				iter_i += 1
				score = quma.get_reward(x)
				logger.info("iter: {}, loss: {}, mean_reward: {}, max_reward: {},  test_reward: {}, his_reward: {}, entropy: {}".
							format(iter_i, -score, score, score, score, score, 0.0))

		# setting the boundary
		bound = 2
		bnds = tuple((0, bound) for _ in range(args.p))

		if args.measure == 'noise':
			sol = opt.minimize(lambda x: -quma.get_reward_noise(args.noise_level, x, batchsize=args.batch_size), x0,
							   bounds=bnds, tol=1e-6,  method=args.qaoa_method, options={'disp': True}, callback=mycallback)

		elif args.measure == 'quantum':
			sol = opt.minimize(lambda x: -quma.get_reward_quantum(x, batchsize=args.batch_size), x0, bounds=bnds,
							   tol=1e-6,  method=args.qaoa_method, options={'disp': True}, callback=mycallback)

		elif args.measure == 'plain':
			sol = opt.minimize(lambda x: -quma.get_reward(x), x0, bounds=bnds,
							   tol=1e-8,  method=args.qaoa_method, options={'disp': True}, callback=mycallback)

		print('==='*10)
		print('Final result: ')
		print('Protocol: ', sol.x)
		print('Duration: ', sum(sol.x))
		print('Fidelity: ', quma.get_reward(sol.x))
		print('==='*10)


	elif args.approach == 'brs':
		protocol = np.random.normal(loc=0.5, scale=0.1, size=args.p)
		t = trange(args.num_iters, desc='Reward')

		for iter_i in t:

			N = np.random.randn(args.npop, args.p)
			WN_plus = args.sigma * N + protocol
			WN_neg = -args.sigma * N + protocol

			if args.measure == 'noise':
				reward_plus = Parallel(n_jobs=num_cpus)(delayed(quma.get_reward_noise)(args.noise_level,
																					   i) for i in WN_plus)
				reward_neg = Parallel(n_jobs=num_cpus)(delayed(quma.get_reward_noise)(args.noise_level,
																					  i) for i in WN_neg)
			elif args.measure == 'quantum':
				reward_plus = Parallel(n_jobs=num_cpus)(delayed(quma.get_reward_quantum)(
					i) for i in WN_plus)
				reward_neg = Parallel(n_jobs=num_cpus)(delayed(quma.get_reward_quantum)(
					i) for i in WN_neg)
			elif args.measure == 'plain':
				reward_plus = Parallel(n_jobs=num_cpus)(delayed(quma.get_reward)(
					i) for i in WN_plus)
				reward_neg = Parallel(n_jobs=num_cpus)(delayed(quma.get_reward)(
					i) for i in WN_neg)

			reward_plus = np.array(reward_plus)
			reward_neg = np.array(reward_neg)

			protocol = protocol + args.alpha_rs / \
				(args.npop) * np.dot(N.T, reward_plus - reward_neg)

			score = quma.get_reward(protocol)
			print('protocol: {}, reward: {}'.format(protocol, score))
			logger.info("iter: {}, loss: {}, mean_reward: {}, max_reward: {},  test_reward: {}, his_reward: {}, entropy: {}".
						format(iter_i, -score, score, score, score, score, 0.0))

			t.set_description('Reward: %g' % score)

	elif args.approach == 'ars':
		protocol = np.random.normal(loc=0.5, scale=0.1, size=args.p)

		t = trange(args.num_iters, desc='Reward')

		for iter_i in t:

			N = np.random.randn(args.npop, args.p)
			WN_plus = args.sigma * N + protocol
			WN_neg = -args.sigma * N + protocol

			if args.measure == 'noise':
				reward_plus = Parallel(n_jobs=num_cpus)(delayed(quma.get_reward_noise)(args.noise_level,
																					   i) for i in WN_plus)
				reward_neg = Parallel(n_jobs=num_cpus)(delayed(quma.get_reward_noise)(args.noise_level,
																					  i) for i in WN_neg)
			elif args.measure == 'quantum':
				reward_plus = Parallel(n_jobs=num_cpus)(delayed(quma.get_reward_quantum)(
					i) for i in WN_plus)
				reward_neg = Parallel(n_jobs=num_cpus)(delayed(quma.get_reward_quantum)(
					i) for i in WN_neg)
			elif args.measure == 'plain':
				reward_plus = Parallel(n_jobs=num_cpus)(delayed(quma.get_reward)(
					i) for i in WN_plus)
				reward_neg = Parallel(n_jobs=num_cpus)(delayed(quma.get_reward)(
					i) for i in WN_neg)

			reward_plus = np.array(reward_plus)
			reward_neg = np.array(reward_neg)

			reward_max = np.maximum(reward_neg, reward_plus)

			ind = np.argsort(-reward_max)
			ind = ind[:args.bpop]

			std = np.std([reward_neg[ind], reward_plus[ind]])

			protocol = protocol + args.alpha_rs / \
				(args.bpop * std) * np.dot(N[ind].T,
										   reward_plus[ind] - reward_neg[ind])

			score = quma.get_reward(protocol)
			print('protocol: {}, reward: {}'.format(protocol, score))
			logger.info("iter: {}, loss: {}, mean_reward: {}, max_reward: {},  test_reward: {}, his_reward: {}, entropy: {}".
						format(iter_i, -score, score, score, score, score, 0.0))

			t.set_description('Reward: %g' % score)

	elif args.approach == 'blackbox':
		# black-box optimization
		names = ["RandomSearch", "CMA", "PSO"]

		bound = 2
		for name in names:
			# budget means the allowed evaluations
			optimizer = ng.optimizers.registry[name](instrumentation=ng.Instrumentation(
				ng.var.Array(args.p).bounded(0, bound)), budget=args.num_iters)

			if args.measure == 'noise':
				recommendation = optimizer.minimize(
					lambda x: -quma.get_reward_noise(args.noise_level, x, batchsize=args.batch_size))

			elif args.measure == 'quantum':
				recommendation = optimizer.minimize(
					lambda x: -quma.get_reward_quantum(x, batchsize=args.batch_size))

			elif args.measure == 'plain':
				recommendation = optimizer.minimize(
					lambda x: -quma.get_reward(x))

			print()
			print(name)
			print(recommendation.args[0])  # optimal args and kwargs
			print(quma.get_reward(recommendation.args[0]))
			logger.info('score: {}.'.format(
				quma.get_reward(recommendation.args[0])))

	elif args.approach == 'pg':
		# control memory for gpu
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		global_step = tf.Variable(0, trainable=False)

		# no learning rate decay
		learning_rate = args.learning_rate
		if args.lr_decay:
			learning_rate = tf.train.exponential_decay(learning_rate, global_step,
													   args.decay_steps, 0.96, staircase=True)

		# optimizer
		if args.optimizer == 'adam':
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		elif args.optimizer == 'sgd':
			optimizer = tf.train.GradientDescentOptimizer(
				learning_rate=learning_rate)
		elif args.optimizer == 'rms':
			optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
		else:
			raise Exception(
				'Optimizer {} is unknown, please choose from the valid categories'.format(args.optimizer))

		model_name = exp_dir + '/controller.ckpt'

		init_val = np.random.randn(args.p)
		init_val = np.expand_dims(init_val, axis=0)

		reinforce = Reinforce(sess, init_val, optimizer, global_step, args)

		saver = tf.train.Saver()

		history_best_protocol = None
		history_best_fid = 0.

		if args.loadmodel_dir:
			saver.restore(sess,  args.loadmodel_dir + '/controller.ckpt')

		t = trange(args.num_iters, desc='Reward')
		for iter_i in t:
			# take action

			action = reinforce.get_action(is_train=True)
			action_flat = np.reshape(action, [args.batch_size, -1])

			reward = Parallel(n_jobs=num_cpus)(
				delayed(quma.get_reward)(i) for i in action_flat)

			
			max_ind = np.argmax(reward)
			if reward[max_ind] > history_best_fid:
				history_best_protocol = action[max_ind]
				history_best_fid = reward[max_ind]

				save_path = saver.save(sess, model_name)
				print("Model saved in path: %s" % (save_path))

				print('==='*10)
				print('History Best: ')
				print('Protocol: ', history_best_protocol)
				print('Fidelity: ', history_best_fid)
				print('==='*10)

			# In our sample action is equal to state
			state = action

			# testing
			test_action = reinforce.get_action(is_train=False)

			test_action = test_action[0]
			test_action_new = np.reshape(test_action, [-1])

			test_reward = quma.get_reward(test_action_new)

			print('Testing ... ')
			print('Test protocol: ', test_action_new)
			print('Fidelity: ', test_reward)

			logger_protocol.info('iter: {}, protocol duration: {}, fid: {}'.format(
				iter_i, test_action_new, test_reward))

			# train and update the policy
			ls, ent = reinforce.train_step(state, reward)


			logger.info("iter: {}, loss: {}, mean_reward: {}, max_reward: {}, test_reward: {}, his_reward: {}, entropy: {}".
						format(iter_i, ls, np.mean(reward), np.max(reward), test_reward, history_best_fid, ent))

			t.set_description('Reward: %g' % np.mean(reward))

		print('==='*10)
		print('History Best: ')
		print('Protocol: ', history_best_protocol)
		print('Fidelity: ', history_best_fid)
		print('==='*10)

	elif args.approach == 'scp':

		def get_cons(x, theta): #x is the actual theta and theta is actually theta tilde (increment)
			'''
			Constrain
			'''
			global args
			cons = [x + theta >= 0, cp.sum(x + theta) == args.T_tot]
			return cons
		if args.testcase == 'XXnYY':
		#Initialization with the two Hamiltonians being applied for equal durations.
			each_T = args.T_tot/2
			protocol_a = np.random.uniform(size = args.p)
			protocol_a /= np.sum(protocol_a)
			protocol_a *= each_T
			protocol_b = np.random.uniform(size = args.p)
			protocol_b /= np.sum(protocol_b)
			protocol_b *= each_T
			protol_init = np.array([item for pair in zip(protocol_a, protocol_b + [0]) 
							for item in pair])
		else:
			protol_init = np.random.rand(args.p*2)
			protol_init /= np.sum(protol_init)
			protol_init *= args.T_tot

		print("The initial fidelity is")
		print(quma.get_reward(protol_init))

		protocol_opt, fid_opt = scp.scpsolve(lambda protocol, delta: quma.get_reward(protocol), 
										protol_init, [0], get_cons, args.fid_target)
		
		logger_protocol.info('iter: {}, protocol duration: {}, fid: {}'.format(
				0, protocol_opt, fid_opt))

		print('==='*10)
		print("After optimization:")
		print('Protocol: ', protocol_opt)
		print('Fidelity: ', fid_opt)
		print('==='*10)


def main():
	global args
	args = parse_args()

	data_dir = os.path.join(args.path, 'exp')

	now = datetime.datetime.now(dateutil.tz.tzlocal())
	# timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
	# exp_name = '{0}_{1}'.format('qrl',
	#                             timestamp)
	exp_name = args.exp_name #Abandoned time naming
	exp_dir = os.path.join(data_dir, exp_name)

	assert not os.path.exists(exp_dir),\
		'Experiment directory {0} already exists. Either delete the directory, or run the experiment with a different name'.format(
			exp_dir)
	os.makedirs(exp_dir, exist_ok=True)

	processes = []

	for e in range(args.n_experiments):
		seed = args.seed + 10*e
		print('Running experiment with seed %d' % seed)

		def train_func():
			train(seed, os.path.join(exp_dir, '%d' % seed))

		p = Process(target=train_func, args=tuple())
		p.start()
		processes.append(p)

	for p in processes:
		p.join()


if __name__ == '__main__':
	main()
