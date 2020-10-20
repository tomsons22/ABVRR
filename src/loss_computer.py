import torch
import torch as tt
import torch.distributions as ttd
from IPython import embed
from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np
from weigth_estimator import WeightEstimator
from cv_builder import CVBuilderH, CVBuilderHVP
from timeit import default_timer as timer


def loss(P, Q, N = 10, seed = 1):
	tt.manual_seed(seed)
	Q.make_dist(N)
	Z = Q.sample_rep()
	log_P = P.log_prob(Z)
	H = Q.entropy()
	loss = log_P + H
	return loss

def loss_gradient(P, Q, N = 10, seed = 1):
	tt.manual_seed(seed)
	Q.make_dist(N)
	Z = Q.sample_rep()
	Zc = Z.detach()
	Zc = Zc.requires_grad_()
	log_Pc = P.log_prob(Zc).sum()

	grad_z_log_P = tt.autograd.grad(log_Pc, Zc)
	grads_w_log_P = tt.autograd.grad(Z, Q.parameters(), grad_outputs = grad_z_log_P, retain_graph = True)
	
	H = Q.entropy().sum()
	grads_w_H = tt.autograd.grad(H, Q.parameters(), allow_unused = True, retain_graph = True)
	# Note: the first thing is zero, the entropy does not depend on the mean
	grads_w_H = [g if g is not None else 0 for g in grads_w_H]
	
	return [gp + gh for (gp, gh) in zip(grads_w_log_P, grads_w_H)], Z, grad_z_log_P[0]





class GradientComputer():
	def __init__(self, dim, modeQ, mode = 'nocv', rank_cv = None):
		"""
		- Possible modes:
			- 'nocv': plain gradient. Available for all Q.
			- 'cv-HVP': CV built using the approximation by Miller et al.
			- 'cv-BFGS': CV built using the learned quadratic expansion.
		"""
		self.mode = mode
		if mode == 'nocv':
			return
		self.w_calc = WeightEstimator()
		if mode == 'cv-HVP' or mode == 'cv-HVP-1':
			self.cvbuilder = CVBuilderHVP()
		else:
			self.cvbuilder = CVBuilderH(dim = dim, modeQ = modeQ, mode = mode, rank_cv = rank_cv)

	def get_cv_weight(self):
		if self.mode == 'cv-HVP-1':
			return 1
		if self.mode == 'nocv':
			return None
		return self.w_calc.get_weight()

	def get_gradient(self, P, Q, N = 10, seed = 1, update = True, cancelcv = False):
		grads, Zc, gc = loss_gradient(P, Q, N, seed)
		if self.mode == 'nocv':
			return grads
		cvs, z0 = self.cvbuilder.getCV(P, Q, N, seed, Zc) # In both cases this has a significant cost
		a = self.get_cv_weight()
		if update:
			self.update(grads, cvs, Zc, gc, z0, Q.covariance()) # For F this is something. For D it's basically nothing.
		return [g + a * c for (g, c) in zip(grads, cvs)]

	def update(self, grads, cvs, Zc, gc, z0, cov = None):
		"""
		- grads is a list of matrices. Each matrix contains a gradient evaluation as rows. Have to flatten it.
		- cvs is a list of matrices. Each matrix contains a cv evaluation as row. Have to flatten it.
		- Zc is a matrix containing the samples z sampled.
		- gc is a matrix containing the gradients of P.log_prob in each of the samples from Zc.
		- z0 is a vector that contains the mean (detached).
		"""
		self.w_calc.update_estimations(grads, cvs)
		if 'cv-quad' in self.mode:
			self.cvbuilder.update_H(Zc, gc, z0, cov)











