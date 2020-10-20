import torch
import torch as tt
import torch.distributions as ttd
from IPython import embed
from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np


def load_Q(d, dim):
	if d == 'diag':
		return DiagGaussian(dim)
	if d == 'full':
		return FullGaussian(dim)
	if d == 'diaglr':
		return DiagLRGaussian(dim, rank = 10)
	if d == 'diaghh':
		return DiagHH(dim)
	raise NotImplementedError('Q %s not implemented.' % d)


class Gaussian():
	def __init__(self, dim, name = 'name'):
		self.dim = dim
		self.name = name

	def parameters(self):
		raise NotImplementedError

	def covariance(self):
		raise NotImplementedError

	def make_dist(self, N):
		raise NotImplementedError

	def batchify(self, N):
		raise NotImplementedError

	def sample_rep(self):
		return self.dist.rsample()

	def entropy(self):
		return self.dist.entropy()

	def log_prob(self, z):
		return self.dist.log_prob(z)

	def _get_params(self):
		raise NotImplementedError

	def _transform_params(self, params):
		raise NotImplementedError

	def project(self):
		return


class DiagGaussian(Gaussian):
	def __init__(self, dim, name = 'diag'):
		super().__init__(dim, name)
		self.mean = torch.zeros(self.dim)
		self.logdiag = torch.zeros(self.dim)
		self.meanb = None
		self.logdiagb = None

	def make_dist(self, N = 1):
		self.batchify(N = N)
		self.dist = ttd.Independent(ttd.Normal(loc = self.meanb, scale = self.logdiagb.exp(), validate_args = True), 1)

	def batchify(self, N):
		mean = self.mean.clone()
		logdiag = self.logdiag.clone()
		self.meanb = mean.view(1, self.dim).repeat(N, 1).requires_grad_()
		self.logdiagb = logdiag.view(1, self.dim).repeat(N, 1).requires_grad_()

	def parameters(self, orig = False):
		if orig:
			return [self.mean, self.logdiag]
		return [self.meanb, self.logdiagb]

	def _get_params(self):
		mean = self.mean.detach().clone().requires_grad_()
		logdiag = self.logdiag.detach().clone().requires_grad_()
		return [mean, logdiag]

	def _transform_params(self, params):
		mean = params[0]
		logdiag = params[1]
		# return mean, logdiag.exp()
		return mean, (2 * logdiag).exp() # This is the covariance

	def covariance(self):
		# return tt.diag((2 * self.logdiag).exp()).detach()
		return (2 * self.logdiag).exp().detach()



class FullGaussian(Gaussian):
	def __init__(self, dim, name = 'full'):
		super().__init__(dim, name)
		self.mean = torch.zeros(self.dim)
		self.M = torch.eye(self.dim)
		self.meanb = None
		self.Mb = None

	def make_dist(self, N = 1):
		self.batchify(N = N)
		self.dist = ttd.MultivariateNormal(loc = self.meanb, scale_tril = tt.tril(self.Mb), validate_args = True)

	def batchify(self, N):
		mean = self.mean.clone()
		M = self.M.clone()
		self.meanb = mean.view(1, self.dim).repeat(N, 1).requires_grad_()
		self.Mb = M.view(1, self.dim, self.dim).repeat(N, 1, 1).requires_grad_()

	def parameters(self, orig = False):
		if orig:
			return [self.mean, self.M]
		return [self.meanb, self.Mb]

	def _get_params(self):
		mean = self.mean.detach().clone().requires_grad_()
		M = self.M.detach().clone().requires_grad_()
		return [mean, tt.tril(M)]

	def _transform_params(self, params):
		mean = params[0]
		M = params[1]
		M_aux = tt.tril(M)
		# return mean, tt.tril(M)
		return mean, M_aux @ M_aux.t() # Again, covariance

	def covariance(self):
		M_aux = tt.tril(self.M).detach()
		return M_aux @ M_aux.t()

	def project(self):
		# Diagonal elements project to be positive
		eps = 1e-6
		for i in range(self.M.shape[0]):
			if self.M[i, i] <= 0:
				self.M[i, i] = eps



class DiagLRGaussian(Gaussian):
	def __init__(self, dim, rank = 5, name = 'diaglr'):
		super().__init__(dim, name)
		self.rank = rank
		self.mean = torch.zeros(self.dim)
		self.logdiag = torch.zeros(self.dim)
		self.F = torch.zeros(self.dim, rank)
		self.meanb = None
		self.logdiagb = None
		self.Fb = None

	def make_dist(self, N = 1):
		self.batchify(N = N)
		self.dist = ttd.LowRankMultivariateNormal(loc = self.meanb, cov_diag = (self.logdiagb * 2).exp(),
			cov_factor = self.Fb, validate_args = True)

	def batchify(self, N):
		mean = self.mean.clone()
		logdiag = self.logdiag.clone()
		F = self.F.clone()
		self.meanb = mean.view(1, self.dim).repeat(N, 1).requires_grad_()
		self.logdiagb = logdiag.view(1, self.dim).repeat(N, 1).requires_grad_()
		self.Fb = F.view(1, self.dim, self.rank).repeat(N, 1, 1).requires_grad_()

	def parameters(self, orig = False):
		if orig:
			return [self.mean, self.logdiag, self.F]
		return [self.meanb, self.logdiagb, self.Fb]

	def _get_params(self):
		mean = self.mean.detach().clone().requires_grad_()
		logdiag = self.logdiag.detach().clone().requires_grad_()
		F = self.F.detach().clone().requires_grad_()
		return [mean, logdiag, F]

	def _transform_params(self, params):
		mean = params[0]
		logdiag = params[1]
		F = params[2]
		return mean, tt.diag((2 * logdiag).exp()) + F @ F.t() # This is the covariance

	def covariance(self):
		C = tt.diag((2 * self.logdiag).exp()) + self.F @ self.F.t()
		return C.detach()


