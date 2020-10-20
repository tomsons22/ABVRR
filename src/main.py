import torch
import torch as tt
import torch.distributions as ttd
from IPython import embed
from tqdm import tqdm
import numpy as np
from model import load_model
from q_distr import load_Q
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from loss_computer import GradientComputer, loss
from timeit import default_timer as timer
import argparse
import pickle
import sys


# PARSE ARGUMENTS
args_parser = argparse.ArgumentParser(description = 'Process arguments to run')
args_parser.add_argument('-limit', type = str, default = 'i', help = 'Run with time(t) / iters(i) limit.')
args_parser.add_argument('-iters', type = int, default = 1500, help = 'Number of iterations.')
args_parser.add_argument('-time', type = int, default = 5, help = 'Optimization time, in seconds.')
args_parser.add_argument('-Q', type = str, default = 'diag', help = 'Q distribution: diag, full.')
args_parser.add_argument('-model', type = str, default = 'logistic-sonar', help = 'Model p to use.')
args_parser.add_argument('-N', type = int, default = 5, help = 'Samples to estimate gradient.')
args_parser.add_argument('-reps', type = int, default = 1, help = 'Number of repetitions.')
args_parser.add_argument('-run_cluster', type = int, default = 0, help = '1: true, 0: false.')
args_parser.add_argument('-lr', type = float, default = 0.05, help = 'Learning rate.')
args_parser.add_argument('-id', type = int, default = -1, help = 'ID.')
args_parser.add_argument('-seed', type = int, default = 1, help = 'Random seed to use.')
args_parser.add_argument('-opt', type = str, default = 'Adam', help = 'Optimizer to use: Adam or SGDM')
run_info = vars(args_parser.parse_args())

Qd = run_info['Q']
model = run_info['model']
N = run_info['N']
reps = run_info['reps']
lr = run_info['lr']
ID = run_info['id']
seed_np = run_info['seed']

modescv = ['cv-quad', 'nocv', 'cv-HVP'] # [ours, no CV, Taylor]

# Performs optimization
def run():
	def done():
		if run_info['limit'] == 'i':
			if iters >= run_info['iters']:
				return True
			return False
		if run_info['limit'] == 't':
			if opt_time >= run_info['time']:
				return True
			return False
		raise NotImplementedError('Unrecognized limit condition.')

	def get_opt():
		if run_info['opt'] == 'SGDM':
			return SGD(Q.parameters(orig = True), lr = lr, momentum = 0.9)
		if run_info['opt'] == 'Adam':
			return Adam(Q.parameters(orig = True), lr = lr, amsgrad = False)

	P = load_model(model)
	Q = load_Q(Qd, P.dim)
	gc = GradientComputer(dim = P.dim, modeQ = Qd, mode = modecv)
	opt = get_opt()
	params_orig = Q.parameters(orig = True)
	losses = []
	weights = []
	opt_times = []
	iters = 0
	opt_time = 0
	while not done():
		# Update a few things
		seed = base_seed + iters
		# Print every now and then
		if (iters + 1) % 500 == 0:
			try:
				print('Optimizing, iteration %i, weight %.7f.' % (iters, w))
			except:
				print('Optimizing, iteration %i.' % (iters))
		# Track training ELBO
		if iters % factor_track == 0:
			opt_times.append(opt_time)
			loss_ = loss(P, Q, N_track, seed).mean()
			losses.append(loss_.item())
			w = gc.get_cv_weight()
			weights.append(w)
		# Compute gradient and perform optimization step
		start = timer()
		grads = gc.get_gradient(P, Q, N = N, seed = seed, update = True)
		for p, g in zip(params_orig, grads):
			p.grad = -g.mean(0)
		opt.step()
		opt.zero_grad()
		Q.project()
		opt_time += (timer() - start)
		iters += 1
	results = {'ELBO': losses, 'cvweigths': weights, 'times': opt_times}
	return results


# Get seed
np.random.seed(seed_np)
base_seed = np.random.randint(1, 10000)
# Set cores for cluster
tt.set_num_threads(2)
# Set tracking
N_track = 500 # 200
factor_track = 10 # 50

# Run reps times
results_all = []
for r in range(reps):
	base_seed = np.random.randint(1, 10000)
	for modecv in modescv:
		print(modecv)
		run_info['modecv'] = modecv
		ID += 1 # Has to be unique, careful here when sending jobs to cluster
		results = run()
		run_info['results'] = results
		results_all.append(results)

plt.figure()
for r, l in zip(results_all, modescv):
	ELBO = r['ELBO']
	print(ELBO[-1])
	times = r['times']
	if run_info['limit'] == 't':
		plt.plot(times, ELBO, label = l)
	else:
		plt.plot(ELBO, label = l)
plt.legend(loc = 0)
plt.show()









