#-------------------------------------------------------------------------
# Solving robust optimization via sequential convex programming
#
# Package dependence:
#           python 3.7
#           cvxpy
#           numpy
#
# The method is designed for the following problem:
#
#                max_{a} f^*(a),
#           where f^*(a) := min_{b\in\delta} f(a,b)
#           subject to   a\in\Theta
#
# Input:
#   fidelity --- Objective function f(a,b) 
#                scpsolve:          evaluate grad, hess via numerical differentiate
#                scpsolve_batch:    grad, hess must be explicitly given
#       grad --- Gradient of obj function
#       hess --- Hessian of obj function
#      theta --- Initial value
#      delta --- Sample noise
#       opts --- Options structure with fields
#              printstep: print frequency
#              ared_eps: stop criteria of increment
#              epsgrad: step size of numerical differentiate
#              gamma1,gamma2,eta1,eta2,eta3: para used to update turst region
#              trust_init: initial trust region
#              uppertrust: max trust region
#              lowertrust: min trust region
# Default: use cvx to solve 2-order problem
# Subproblem: SDP
# Return before optimum:
#       small increment --- increment is small
#       stuck - k       --- solver stucks with period (k + 1)
#
#
# Output:
#       theta --- Solution of robust optimization problem
#   obj_value --- The robust obj value f^*(theta)
#
#-------------------------------------------------------------------------
#
# Author: Y. Dong
# Version 1.1.0
# Last Update 09/2019
#
# Modified by Zhibo Yang for python3 compatability
#-------------------------------------------------------------------------


import numpy as np
from functools import reduce
import cvxpy as cp
#from mpi4py import MPI

class SCPConfig(object):
    trust_init = 0.1
    eta1, eta2, gamma1, gamma2, uppertrust, epstrust = 0.25, 0.75, 0.25, 2.0, 100.0, 1e-5
    lowertrust = 1e-5
    eta3 = 2.0

def NSDSqrt(mat):
    val, vec = np.linalg.eig(mat)
    val[val > 0] = 0
    return vec.dot(np.diag(np.sqrt(-val)))

def convexQP(
    fid, grad, nsdsqrt,  # [fidelities, gradients, hessians]
    theta,      # current theta
    trust,      # current trust region
    get_cons    # function(x, theta) that returns constraint
    ):
    nsample, controlsize = grad.shape
    # construction of cvx problem
    f0 = cp.Variable()
    x = cp.Variable(controlsize)    # theta tilde
    cons = [cp.bmat([[cp.reshape(fid[i] - f0 + x @ grad[i, :], (1, 1)), cp.reshape(x.T @ nsdsqrt[i] / np.sqrt(2), (1, controlsize))],
            [cp.reshape(nsdsqrt[i].transpose() @ x / np.sqrt(2), (controlsize, 1)), np.identity(controlsize)]]) >> 0 for i in range(nsample) ]
    cons = cons + [cp.norm(x, 2) <= trust]
    cons = cons + get_cons(x, theta)
    obj = cp.Maximize(f0)
    prob = cp.Problem(obj, cons)
    prob.solve()
    return x.value

#Compute the gradient
def gradfid(fidelity, gradfid_pos, gradfid_omega, epsgrad = 1e-5):
    basis = np.identity(gradfid_pos.size)
    grad_1 = np.array([fidelity(gradfid_pos + epsgrad * basis[:, i], gradfid_omega) for i in range(gradfid_pos.size)])
    grad_2 = np.array([fidelity(gradfid_pos - epsgrad * basis[:, i], gradfid_omega) for i in range(gradfid_pos.size)])
    return (grad_1 - grad_2) / (2 * epsgrad)

#Compute the Hessian
def scndordfid(fidelity, scnd_pos, scnd_omega, epsgrad = 1e-5):
    ret = np.zeros([scnd_pos.size, scnd_pos.size])
    for k in range(scnd_pos.size):
        for j in range(k):
            perturb = [np.zeros(scnd_pos.size) for i in range(4)]
            perturb[0][k], perturb[0][j] = epsgrad, epsgrad
            perturb[1][k], perturb[1][j] = epsgrad, -epsgrad
            perturb[2][k], perturb[2][j] = -epsgrad, epsgrad
            perturb[3][k], perturb[3][j] = -epsgrad, -epsgrad
            fid = [fidelity(scnd_pos+p, scnd_omega) for p in perturb]
            ret[k, j] = ( fid[0] - fid[1] - fid[2] + fid[3] ) / (2 * epsgrad)**2
            ret[j, k] = ret[k, j]
        perturb = [np.zeros(scnd_pos.size) for i in range(3)]
        perturb[0][k] = epsgrad
        perturb[2][k] = -epsgrad
        fid = [fidelity(scnd_pos+p, scnd_omega) for p in perturb]
        ret[k, k] = ( fid[0] - 2 * fid[1] + fid[2] ) / epsgrad**2
    return ret

def batchfidelity_mpi(fidelity, batch_theta, batch_delta, batch_gradients = False):
    pass

def batchfidelity(fidelity, batch_theta, batch_delta, batch_gradients = False):
    if batch_gradients:
        fiddata = [[fidelity(batch_theta, omega), gradfid(fidelity, batch_theta, omega), scndordfid(fidelity, batch_theta, omega)] for omega in batch_delta]
        fid = np.array([p[0] for p in fiddata])
        grad = np.array([p[1].tolist() for p in fiddata])
        hess = [p[2] for p in fiddata]
        return fid, grad, hess
    else:
        fid = [fidelity(batch_theta, omega) for omega in batch_delta]
        return np.array(fid)

def affineLP(
    fid, fid_grad,  # fidelities and gradients
    trust      # current trust region
    ):
    nsample, controlsize = fid_grad.shape
    # construction of cvx problem
    f0 = cp.Variable()
    fi = cp.Parameter(nsample, value = fid)  # fidelity
    gi = cp.Parameter((nsample, controlsize), value = fid_grad)   # each row is a gradient
    x = cp.Variable(controlsize)    # theta tilde
    cons = [fi + gi @ x >= f0] + [cp.norm(x, 2) <= trust]
    obj = cp.Maximize(f0)
    prob = cp.Problem(obj, cons)
    prob.solve()
#    print (np.linalg.norm(x.value), trust)
    return x.value

def scpsolve(fidelity, theta_init, scp_delta, get_cons, targ_fid = 0.999, trust_init = 0, ared_eps = 1e-5, printstep = 10):
    config = SCPConfig()
    if trust_init == 0:
        trust = config.trust_init
    elif trust_init > 0:
        trust = trust_init
    else:
        raise 'invalid trust_init, use that in config'
        trust = config.trust_init
    theta = theta_init
    count, fid, ared = 0, 0.0, 1.0
    update = True
    update_theta = True
    ared_0, ared_1, ared_2, trust_0, trust_1, trust_2 = 0, 1, 2, 0, 1, 2
    while np.min(fid) < targ_fid and np.abs(ared) > ared_eps and update:
        count = count + 1
        if update_theta:
            fid, fid_grad, fid_hess = batchfidelity(fidelity, theta, scp_delta, batch_gradients = True)
            nsdsqrt = [NSDSqrt(p) for p in fid_hess]
        thetatilde = convexQP(fid, fid_grad, nsdsqrt, theta, trust, get_cons)
 #       thetatilde = affineLP(fid, fid_grad, theta, trust)
        quad = [p.transpose().dot(thetatilde) for p in nsdsqrt]
        quad_approx = np.array([np.inner(p, p) for p in quad])
        pred = np.min( fid + fid_grad.dot(thetatilde) - 0.5 * quad_approx) - np.min(fid)
        ared = np.min( batchfidelity(fidelity, theta+thetatilde, scp_delta, batch_gradients = False) ) - np.min(fid)
        ratio = ared / pred
        theta, trust, update, update_theta = trust_region_update(ratio, ared, trust, theta, thetatilde, config)
        if printstep != 0 and np.mod(count, printstep) == 0:
            print (np.min(fid), ratio, ared, trust)
#            print -np.log10(1-np.min(fid)), -np.log10(1-np.max(fid))
        if ared_0 == ared and trust_0 == trust:
            print ('stuck - 2')
            break
        if ared_1 == ared and trust_1 == trust:
            print ('stuck - 1')
            break
        ared_0, trust_0 = ared_1, trust_1
        ared_1, trust_1 = ared_2, trust_2
        ared_2, trust_2 = ared, trust
        if count > 10 and np.abs(ared / fid.min()) < 5e-3:
            update = False
            print ('small increment')
    return theta, np.min(fid)

def scpsolve_batch(batch_fidelity_function, theta_init, get_cons, targ_fid = 0.999, trust_init = 0, ared_eps = 1e-5, printstep = 10):
    config = SCPConfig()
    if trust_init == 0:
        trust = config.trust_init
    elif trust_init > 0:
        trust = trust_init
    else:
        raise 'invalid trust_init, use that in config'
        trust = config.trust_init
    theta = theta_init
    count, fid, ared = 0, 0.0, 1.0
    update = True
    update_theta = True
    while np.min(fid) < targ_fid and np.abs(ared) > ared_eps and update:
        count = count + 1
        if update_theta:
            fid, fid_grad, fid_hess = batch_fidelity_function(theta, True)
            nsdsqrt = [NSDSqrt(p) for p in fid_hess]
        thetatilde = convexQP(fid, fid_grad, nsdsqrt, theta, trust, get_cons)
        quad = [p.transpose().dot(thetatilde) for p in nsdsqrt]
        quad_approx = np.array([np.inner(p, p) for p in quad])
        pred = np.min( fid + fid_grad.dot(thetatilde) - 0.5 * quad_approx) - np.min(fid)
        ared = np.min( batch_fidelity_function(theta+thetatilde, False) ) - np.min(fid)
        ratio = ared / pred
        theta, trust, update, update_theta = trust_region_update(ratio, ared, trust, theta, thetatilde, config)
        if printstep != 0 and np.mod(count, printstep) == 0:
            print (np.min(fid), ratio, ared, trust)
#            print -np.log10(1-np.min(fid)), -np.log10(1-np.max(fid))
    return theta, np.min(fid)

def trust_region_update(ratio, ared, trust, theta, thetatilde, config):
    theta_out = theta
    trust_out = trust
    update_out = True
    update_theta = False
    if ratio < config.eta1 or ratio > config.eta3:
        trust_out = config.gamma1 * trust
    else:
        if ared > 0:
            theta_out = theta + thetatilde
            update_theta = True
        elif trust == config.uppertrust:
            update_out = False
        if ratio >= config.eta2:
            trust_out = min(config.gamma2 * trust, config.uppertrust)
        elif ared <= 0:
            update_out = False
    if trust < config.lowertrust:
        update_out = False
    return theta_out, trust_out, update_out, update_theta