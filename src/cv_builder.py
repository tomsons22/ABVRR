import torch
import torch as tt
import torch.distributions as ttd
from IPython import embed
from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np
from quad_approx import QuadApproximator


# Multiple Hessian vector products
def hessianvectormult(f, x, v):
	def fsum(f, x):
		return f(x).sum()
	def grad(f, x):
		return torch.autograd.grad(fsum(f, x), x, create_graph = True)[0]

	N, D = v.shape
	x = torch.clone(x.unsqueeze(0).expand(N, D)).requires_grad_()
	Hv = torch.autograd.grad((grad(f, x) * v).sum(), x, create_graph = False)[0]
	return Hv


def hessianvector(f, x, v):
	def grad(f, x):
		return torch.autograd.grad(f(x), x, create_graph = True)[0]

	x = torch.clone(x).requires_grad_()
	Hv = torch.autograd.grad(grad(f, x) @ v, x, create_graph = False)[0]
	return Hv


def hessian(f, x):
	def grad(f, x):
		return torch.autograd.grad(f(x), x, create_graph = True)[0]
	
	x = torch.clone(x).requires_grad_()
	D = x.shape[0]
	H = torch.zeros(D, D)
	for i in range(D):
		H[i, :] = torch.autograd.grad(grad(f, x)[i], x, create_graph = False)[0]
	return H


def grad(f, x):
	x = torch.clone(x).requires_grad_()
	return torch.autograd.grad(f(x), x, create_graph = False)[0]


class CVBuilderH():
	def __init__(self, dim, modeQ, mode = 'cv-quad', rank_cv = None):
		self.mode = mode
		if rank_cv is None:
			if modeQ == 'diaglr' or modeQ == 'diag':
				self.H_approximator = QuadApproximator(dim = dim, rank = 10)
			if modeQ == 'full':
				self.H_approximator = QuadApproximator(dim = dim, rank = 20)
		else:
			self.H_approximator = QuadApproximator(dim = dim, rank = rank_cv)


	def update_H(self, Zc, gc, z0, cov):
		self.H_approximator.update_params(Zc, gc, z0, cov)


	def getCV(self, P, Q, N, seed, Z):
		tt.manual_seed(seed)
		f = P.log_prob
		z0 = Q.mean.detach()
		
		fhat = self.H_approximator.eval_quadratic(Z, z0).sum()
		grads_hat = tt.autograd.grad(fhat, Q.parameters(), retain_graph = True) # Each has shape (M, dim)
		
		# Compute expectation of f
		params = Q._get_params()

		# This is one way (not using structures -- less efficient for bigger models)
		# mean, cov = Q._transform_params(params)
		# Efhat = self.H_approximator.eval_quadratic_exp_bis(mean, cov, z0)
		
		# This is the other way (using structure)
		Efhat = self.H_approximator.eval_quadratic_exp(params, Q.name, z0)
		
		grads_E = tt.autograd.grad(Efhat, params, retain_graph = True)
		
		# Build CV
		cv = [ge - gh for (ge, gh) in zip(grads_E, grads_hat)]
		return cv, z0





class CVBuilderHVP():
	def getCV(self, P, Q, N, seed, Z):
		if Q.name == 'diag':
			return self.getCV_d(P, Q, N, seed)
		if Q.name == 'full':
			return self.getCV_f(P, Q, N, seed)
		if Q.name == 'diaglr':
			return self.getCV_dlr(P, Q, N, seed)
		raise NotImplementerError('HVP method not implemented for %s' % Q.name)


	def getCV_dlr(self, P, Q, N, seed):
		# This is the resulting thing without baselines that cancel out
		tt.manual_seed(seed)
		dim = P.dim
		f = P.log_prob
		z0 = Q.mean.detach()
		diag = Q.logdiag.detach().exp()
		F = Q.F.detach()
		rank = F.shape[1]
		g0 = grad(f, z0) # (dim)
		eps_f = tt.randn(N, rank)
		eps_d = tt.randn(N, dim)

		Seps_d = diag * eps_d
		Seps_F = (F @ eps_f.t()).t()
		hvps = hessianvectormult(f, z0, Seps_d + Seps_F)

		cvdiag = -g0 * Seps_d

		cvF = -tt.einsum('j, ik -> ijk', g0, eps_f)

		return (-hvps, cvdiag, cvF), z0 # ((N, dim), (N, dim), (N, dim, rank))


	def getCV_d(self, P, Q, N, seed):
		# This is the resulting thing without baselines that cancel out
		tt.manual_seed(seed)
		dim = P.dim
		f = P.log_prob
		z0 = Q.mean.detach()
		sigma = Q.logdiag.detach().exp()
		g0 = grad(f, z0) # (dim)
		eps = tt.randn(N, dim) # (N, dim)

		Seps = sigma * eps # (N, dim)
		hvps = hessianvectormult(f, z0, Seps) # (N, dim)

		return (-hvps, -g0 * Seps), z0 # ((N, dim), (N, dim))

	def getCV_f(self, P, Q, N, seed):
		# This is the resulting thing without baselines that cancel out
		tt.manual_seed(seed)
		dim = P.dim
		f = P.log_prob
		z0 = Q.mean.detach()
		M = Q.M.detach()
		g0 = grad(f, z0) # (dim)
		eps = tt.randn(N, dim) # (N, dim)

		Seps = (M @ eps.t()).t() # (N, dim)
		hvps = hessianvectormult(f, z0, Seps) # (N, dim)

		cvscale = -tt.einsum('j, ik -> ijk', g0, eps)

		return (-hvps, cvscale), z0 # ((N, dim), (N, dim, dim))




# class CVBuilderHVP():
# 	def getCV(self, P, Q, N, seed):
# 		dim = P.dim
# 		mean = Q.mean.detach()
# 		diag = Q.logdiag.detach().exp()
# 		f = P.log_prob
# 		tt.manual_seed(seed)
# 		eps = tt.randn(N, dim) # (N, dim)
# 		eps_scaled = eps * diag # (N, dim)
# 		z0 = mean.detach().clone() # (dim)
# 		start = timer()
# 		hvps = hessianvectormult(f, z0, eps_scaled) # (N, dim)
# 		# hvps = tt.zeros(N, dim)
# 		gmu = grad(f, z0) # (dim)

# 		dLdz = gmu + hvps # (N, dim)
# 		dLds = dLdz * (eps_scaled) + 1. # (N, dim)

# 		Hdiag_sum = (eps * hvps).sum(0) # (dim)
# 		Hdiag_s = (Hdiag_sum - eps * hvps) / (N - 1.) # (N, dim)
# 		dLds_mu = Hdiag_s * diag + 1. / diag.view(1, -1) * diag # necessary the view? (N, dim)
		
# 		# return (-hvps, dLds_mu - dLds), mean # ((N, dim), (N, dim))
# 		return (-hvps * 0., dLds_mu - dLds), mean # ((N, dim), (N, dim))



# # dLdz    = gmu + hvps
# # dLds    = dLdz * (eps*s_lam) + 1

# # # compute Leave One Out approximate diagonal (per-sample mean of dLds)
# # Hdiag_sum = np.sum(eps*hvps, axis=0)
# # Hdiag_s   = (Hdiag_sum[None,:] - eps*hvps) / float(ns-1)
# # dLds_mu   = (Hdiag_s + 1/s_lam[None,:]) * s_lam




# # class CVBuilderHVP():
# # 	def getCV(self, P, Q, N, seed):
# # 		tt.manual_seed(seed)
# # 		dim = P.dim
# # 		f = P.log_prob
# # 		z0 = Q.mean.detach()
# # 		sigma = Q.logdiag.detach().exp()
# # 		g0 = grad(f, z0) # (dim)
# # 		eps = tt.randn(N, dim) # (N, dim)
		
# # 		Seps = sigma * eps # (N, dim)
# # 		hvps = hessianvectormult(f, z0, Seps) # (N, dim)
# # 		G = g0 + hvps # (N, dim)

# # 		JG = G * Seps # (N, dim) # Or eps, Seps is to fix the jacobian because of logscale parameterization
# # 		# Build baseline
# # 		JG_sum = JG.sum(0) # (dim)
# # 		baseline = (JG_sum - JG) / (N - 1.) # (N, dim)

# # 		return (-hvps, baseline - JG), z0 # ((N, dim), (N, dim, dim))



# class CVBuilderHVPFull():
# 	# This is for full-rank covariance
# 	def getCV(self, P, Q, N, seed):
# 		tt.manual_seed(seed)
# 		dim = P.dim
# 		f = P.log_prob
# 		z0 = Q.mean.detach()
# 		S = Q.M.detach()
# 		g0 = grad(f, z0) # (dim)
# 		eps = tt.randn(N, dim) # (N, dim)
		
# 		Seps = (S @ eps.t()).t() # (N, dim)
# 		hvps = hessianvectormult(f, z0, Seps) # (N, dim)
# 		G = g0 + hvps # (N, dim)

# 		JG = tt.einsum('ij, ik -> ijk', G, eps) # Not summing, just a bunch of outer products - (N, dim, dim)
# 		# Buld baseline
# 		JG_sum = JG.sum(0)
# 		baseline = (JG_sum - JG) / (N - 1.) # (N, dim, dim)

# 		return (-hvps, baseline - JG), mean # ((N, dim), (N, dim, dim))








