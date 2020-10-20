import torch
import torch as tt
import torch.distributions as ttd
from IPython import embed
from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np



class WeightEstimator():
	def __init__(self, eps = 1e-6, gamma = 0.9):
		self.eps = eps
		self.gamma = gamma
		self.Ecg = None
		self.Ecc = None

	def get_weight(self):
		# return 1
		if self.Ecg is None or self.Ecc is None: # or self.Ecc ** 2 < 10e-4
			return 0
		return -1. * self.Ecg / (self.Ecc + self.eps)

	def update_estimations(self, G, C):
		"""
		- Receives non flattened samples. That is:
		- G is a list of matrices. Each matrix contains gradients for some parameter as rows (or element in index 0).
		- C is a list of matrices. Each matrix contains the CV for some parameter as rows (or element in index 0).
		"""
		G, C = self.flatten_samples(G, C)
		self.update_Ecc(C)
		self.update_Ecg(G, C)

	def update_Ecc(self, C, soft = True):
		""""
		- C is a matrix of shape (M, dim), where M is the number of samples and dim the dimention of the CV.
		"""
		Ecc = (C * C).sum(1) # Get the norm of each CV
		Ecc = Ecc.mean()
		if not soft:
			self.Ecc = Ecc
		else:
			if self.Ecc is None:
				self.Ecc = 2 * Ecc
			self.Ecc = self.gamma * self.Ecc + (1 - self.gamma) * Ecc

	def update_Ecg(self, G, C, soft = True):
		Ecg = (G * C).sum(1)
		Ecg = Ecg.mean()
		if not soft:
			self.Ecg = Ecg
		else:
			if self.Ecg is None:
				self.Ecg = 0
			self.Ecg = self.gamma * self.Ecg + (1 - self.gamma) * Ecg

	def flatten_samples(self, G, C):
		"""
		Input
		- G is a list of matrices. Each matrix contains a gradient evaluation as rows (or in position 0). Have to flatten it.
		- C is a list of matrices. Each matrix contains a cv evaluation as row. Have to flatten it.
		Output:
		- G is a matrix that contains all G[0] together as a long vector
		- C is a matrix that contains all C[0] together as a long vector
		"""
		G = [g.flatten(start_dim = 1) for g in G]
		G = tt.cat(G, dim = 1)
		C = [cv.flatten(start_dim = 1) for cv in C]
		C = tt.cat(C, dim = 1)
		return G.detach(), C.detach()