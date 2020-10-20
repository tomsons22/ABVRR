import torch
import torch as tt
import torch.distributions as ttd
from IPython import embed
from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np
from torch.optim import Adam, SGD, RMSprop




class QuadApproximator():
	def __init__(self, dim, opt = 'Adam', rank = 10):
		self.approx = ApproximatorDLR(dim, rank)
		
		if opt == 'Adam':
			self.opt = Adam(self.approx.get_params(), lr = 0.01)
		if opt == 'SGD':
			self.opt = SGD(self.approx.get_params(), lr = 0.001, momentum = 0.9)

	def update_params(self, Z, gZ, mu, cov):
		self.approx.set_gradients(Z, gZ, mu, cov)
		self.opt.step()

	def get_params_approx(self):
		return self.approx.get_params_approx()

	def eval_quadratic(self, Z, z0):
		return self.approx.eval_quadratic(Z, z0)

	def eval_quadratic_exp(self, mean, cov, z0):
		return self.approx.eval_quadratic_exp(mean, cov, z0)

	def eval_quadratic_exp_bis(self, params, Qmode, z0):
		return self.approx.eval_quadratic_exp_bis(params, Qmode, z0)




class ApproximatorDLR():
	def __init__(self, dim, rank):
		self.R = rank
		self.b = tt.zeros(dim, requires_grad = True)
		self.f = tt.ones(1, requires_grad = True)
		self.U = tt.randn(dim, self.R) * 0.001
		self.V = tt.randn(dim, self.R) * 0.001
		self.D = tt.randn(dim) * 0.001
		self.U.requires_grad_()
		self.V.requires_grad_()
		self.D.requires_grad_()

	def get_params(self):
		return self.b, self.D, self.V, self.U, self.f

	def eval_quadratic(self, Z, z0):
		S = Z - z0

		val = (self.b * S).sum(1)
		val = val + self.f.exp() * (S * (self.D * S)).sum(1) / 2.
		val = val + self.f.exp() * (S * (self.U @ (self.V.t() @ S.t())).t()).sum(1) / 2.
		return val

	def eval_quadratic_exp(self, params, Qmode, z0):
		mean = params[0]
		val = self.b @ (mean - z0)
		if Qmode == 'diag':
			logdiag = params[1]
			cov = (logdiag * 2.).exp()
			val = val + self.f.exp() * self.D @ cov / 2.
			val = val + self.f.exp() * ((self.V.t() * cov) * self.U.t()).sum() / 2.
		if Qmode == 'full':
			M = params[1]
			cov = M @ M.t()
			val = val + self.f.exp() * self.D @ tt.diag(cov) / 2.
			val = val + self.f.exp() * (self.V * (cov @ self.U)).sum() / 2.
		if Qmode == 'diaglr':
			logdiag = params[1]
			diag = (logdiag * 2.).exp()
			F = params[2]
			val = val + self.f.exp() * (self.D @ diag) / 2. # D_v D_w
			val = val + self.f.exp() * ((self.V.t() * diag) * self.U.t()).sum() / 2. # D_w UV
			val = val + self.f.exp() * ((F.t() * self.D) * F.t()).sum() / 2. # D_v FF
			val = val + self.f.exp() * ((self.V.t() @ F) * (F.t() @ self.U).t()).sum() / 2.
		return val

	def eval_quadratic_exp_bis(self, mean, cov, z0):
		val = self.b @ (mean - z0)
		if len(cov.shape) == 1:
			val = val + self.f.exp() * self.D @ cov / 2.
			val = val + tt.trace((self.f.exp() * self.V.t() * cov) @ self.U) / 2.
		else:
			val = val + self.f.exp() * self.D @ tt.diag(cov) / 2.
			val = val + tt.trace(self.f.exp() * self.V.t() @ (cov @ self.U)) / 2.
		return val

	def set_gradients(self, Z, gZ, mu, cov):
		M = Z.shape[0]
		S = Z - mu

		auxx = self.b + (self.f.exp() * self.D) * S + ((self.f.exp() * self.U) @ (self.V.t() @ S.t())).t() - gZ # Computing this could be
		# avoided by storing previously computed stuff in eval_quadratic
		loss = (auxx * auxx).sum() / M
		grad_D, grad_U, grad_V, grad_b, grad_f = tt.autograd.grad(loss, [self.D, self.U, self.V, self.b, self.f])
		self.D.grad = grad_D
		self.U.grad = grad_U
		self.V.grad = grad_V
		self.b.grad = grad_b
		self.f.grad = grad_f

