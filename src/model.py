import torch
import torch as tt
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from math import pi
import sklearn.datasets
import sklearn.preprocessing
import scipy.stats as sst
import pandas as pd
from torch.distributions.utils import lazy_property
import json
import torch.distributions as ttd
import matplotlib.pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer




def load_model(model):
	if model == 'linear-breastcancer':
		return LinearRegression('breast')
	if model == 'logistic-sonar':
		return LogisticRegression('sonar')
	if model == 'logistic-a1a':
		return LogisticRegression('a1a')
	if model == 'friskm':
		return FriskM()
	if model == 'bnnrw':
		return BNNRW()
	if model == 'dgauss':
		return DiagGaussian(150)
	raise NotImplementedError('Model %s not valid.' % model)




class Gaussian():
	def __init__(self, dim):
		self.dim = dim
		self.name = 'DG'

	def make_dist(self):
		raise NotImplementedError

	def entropy(self):
		return self.dist.entropy()

	def log_prob(self, z):
		return self.dist.log_prob(z)


class DiagGaussian(Gaussian):
	def __init__(self, dim):
		super().__init__(dim)
		tt.manual_seed(1)
		self.mean = torch.randn(self.dim)
		self.logdiag = torch.randn(self.dim) * 0.4
		self.make_dist()

	def make_dist(self):
		self.dist = ttd.Independent(ttd.Normal(loc = self.mean, scale = self.logdiag.exp(), validate_args = True), 1)

	def covariance(self):
		return torch.diag(self.dist.variance)

	def hessian(self, z0):
		return torch.diag(-1. / self.dist.variance)



def load_breast_cancer():
	# Predicting recurrence time
	r_times = []
	features = []
	with open('./datasets/breast_cancer.data', 'rt') as file:
		for line in file:
			if '?' in line:
				continue # Ignore those with missing attributes
			feats = []
			line = line.split(',')
			line = line[2:]
			r_times.append(float(line[0]))
			line = line[1:]
			for f in line:
				feats.append(float(f))
			features.append(feats)
	return np.array(features), np.array(r_times)

class LinearRegression(tt.distributions.Distribution):
	def __init__(self, dset):
		self.name = 'LinR%s' % dset
		self.dset = dset
		X, Y = self.loadData(dset)
		self.X = torch.from_numpy(X).float()
		self.Y = torch.from_numpy(Y).float()
		N = 500
		if self.X.shape[0] > N:
			self.X = self.X[:N, :]
			self.Y = self.Y[:N]
		self.dim = self.X.size()[1] + 1 # For scale
		self.N = self.X.size()[0]
		self.scale_prior = 1.

	def loadData(self, dset):
		import csv
		def pad_with_const(X):
			extra = np.ones((X.shape[0],1))
			return np.hstack([extra, X])
		if 'breast' in dset:
			X, Y = load_breast_cancer()

		# Normalize data
		mux = np.mean(X, axis = 0)
		sigmax = np.std(X, axis = 0)
		X = (X - mux) / sigmax
		X = pad_with_const(X)
		muy = np.mean(Y)
		sigmay = np.std(Y)
		Y = (Y - muy) / sigmay
		return X, Y

	def name(self):
		return 'Linear' + self.dset

	def unpack(self, z):
		return z[:, :-1], z[:, -1]

	def log_prob(self, z):
		return self.logprior(z) + self.loglikelihood(z)

	def logprior(self, z):
		# z.shape = (a, dim)
		if len(z.shape) == 1:
			dist = tt.distributions.Normal(loc = 0, scale = self.scale_prior)
			return dist.log_prob(z).sum()	
		dist = tt.distributions.Normal(loc = 0, scale = self.scale_prior)
		return dist.log_prob(z).sum(-1)

	def loglikelihood(self, z):
		if len(z.shape) == 1:
			beta = z[:-1]
			lnsigma = z[-1]
			loc = self.X @ beta # shape = (N)
			scale = lnsigma.exp() # shape = (1)
			dist = tt.distributions.Normal(loc = loc, scale = scale) # parameters (N)
			return dist.log_prob(self.Y).sum()
		beta, lnsigma = self.unpack(z)
		loc = beta @ self.X.t() # shape = (a, N). Each row contains the results for each sample z against the whole dataset
		scale = lnsigma.exp().unsqueeze(-1) # shape = (a, 1)
		dist = tt.distributions.Normal(loc = loc, scale = scale) # parameters (a, N)
		return dist.log_prob(self.Y).sum(-1)




class LogisticRegression(tt.distributions.Distribution):
	def __init__(self, dset):
		self.name = 'LR%s' % dset
		self.dset = dset
		X, Y = self.loadData(dset)
		self.X = torch.from_numpy(X).float()
		self.Y = torch.from_numpy(Y).float()
		self.X = self.X[:500, :]
		self.Y = self.Y[:500]
		self.dim = self.X.size()[1]
		self.N = self.X.size()[0]
		self.scale_prior = 1.

	def loadData(self, dset):
		def pad_with_const(X):
			extra = np.ones((X.shape[0],1))
			return np.hstack([extra, X])
		if dset == 'sonar':
			X, Y = sklearn.datasets.load_svmlight_file('./datasets/sonar_scale')
		elif dset == 'a1a':
			X, Y = sklearn.datasets.load_svmlight_file('./datasets/a1a')
		X = pad_with_const(np.array(X.todense()))
		assert(len(Y) == X.shape[0])
		return X, Y

	def event_shape(self):
		return tt.Size([self.dim])

	def name(self):
		return "Logistic" + self.dset

	def logp(self, z):
		return self.log_prob(z)

	def log_prob(self, z, full = None):
		"""
		- Works for z of any size, as long as the last one is dim (which gets killed).
		- If z.size() = (a, b, c, d, dim) then logp.size() = (a, b, c, d).
		"""
		logp = self.logprior(z)
		logp = logp + self.loglikelihood(z)
		return logp

	def logprior_z(self, z):
		return self.logprior(z)

	def logprior(self, z):
		"""
		- Works for z of any size, as long as the last one is dim (which gets killed).
		- If z.size() = (a, b, c, d, dim) then logp.size() = (a, b, c, d).
		- Use as prior standard normal.
		"""
		dist = tt.distributions.Normal(loc = 0, scale = self.scale_prior)
		logp = dist.log_prob(z).sum(-1)
		return logp

	def _logsumexp0(self, A):
		"""
		- Stable way of doing log(1 + exp(A)), elementwise.
		- Works for A of any size.
		"""
		B = torch.zeros(A.size())
		Max = torch.max(A, B)
		return Max + ((B - Max).exp() + (A - Max).exp()).log()

	def loglikelihood(self, z):
		"""
		- Works for z of any size, as long as the last one is dim (which gets killed).
		- If z.size() = (a, b, c, d, dim) then logp.size() = (a, b, c, d).
		"""
		# Fix dimentions for broadcasting
		extra = len(z.size()) - 1
		X = self.X
		Y = self.Y
		if extra != 0:
			X = X.view([X.size()[0]] + [1] * extra + [X.size()[1]])
			Y = Y.view([Y.size()[0]] + [1] * extra)
			z = z.unsqueeze(0)
		# Do rest
		A = X * z
		A = A.sum((-1))
		A = -A * Y
		logp = self._logsumexp0(A)
		logp = -logp.sum(0)
		return logp

	def hessian(self, z):
		H = self.hessian_logprior(z)
		H = H + self.hessian_loglikelihood(z)
		return H

	def hessian_logprior(self, z):
		# For now z should be 1D - easy to generalize if needed
		assert(len(z.shape) == 1)
		D = z.shape[-1]
		return -tt.eye(D)

	def hessian_loglikelihood(self, z):
		# For now z should be 1D - easy to generalize if needed
		assert(len(z.shape) == 1)
		X = self.X # shape (N, D)
		Y = self.Y # shape (N,)
		outer = X.unsqueeze(-1) * X.unsqueeze(1) # shape = (N, D, D). outer[i, :, ;] = outer(X[i, :], X[i, :])
		aux = (X * z).sum(-1)
		aux = aux * Y / 2.
		aux = (aux.exp() + (-1. * aux).exp()).pow(2) # shape = (N)
		aux = -aux.unsqueeze(-1).unsqueeze(-1)
		return (outer / aux).sum(0)









class FriskM():
	def __init__(self):
		self.name = 'Friskm'
		self.df = None
		self.sdf = None
		self.process_dataset()
		self.process_1()
		self.scale_prior = 10.
		self.indices_unpack = {'alpha': [0, 2], 'beta': [2, 34], 'mu': [34, 35], 'lns_a': [35, 36], 'lns_b': [36, 37]}

	def process_dataset(self):
		# Preprocessing code extracted from 
		# "Reducing reparameterization gradient variance"  by Miller et al. (2017)
		df = pd.read_csv('./datasets/frisk_with_noise.dat', skiprows = 6, delim_whitespace = True)
		popdf = df[['pop', 'precinct', 'eth']].groupby(['precinct', 'eth'])['pop'].apply(sum)
		percent_black = np.array([ popdf[i][1] / float(popdf[i].sum()) for i in range(1, 76)] )
		precinct_type = pd.cut(percent_black, [0, .1, .4, 1.])
		df['precinct_type'] = precinct_type.codes[df.precinct.values-1]
		sdf = df[ (df['crime']==2.) & (df['precinct_type']==1) ] # crime = 2., precinct_type = 1, don't change
		self.sdf = sdf
		self.df = df

	def process_1(self):
		sdf = self.sdf
		one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)
		self.precincts = np.sort(np.unique(sdf['precinct']))
		self.Xprecinct = one_hot(sdf['precinct'], 76)[:, self.precincts]
		self.Xeth = one_hot(sdf['eth'], 4)[:, 1:-1]
		self.yep = sdf['stops'].values
		self.lnep = np.log(sdf['past.arrests'].values) + np.log(15./12)
		self.num_eth = self.Xeth.shape[1]
		self.num_precinct = self.Xprecinct.shape[1]
		self.dim = self.num_eth + self.num_precinct + 3

	def _unpack(self, z):
		if len(z.shape) == 1:
			z = z.view(1, -1)
		alpha_eth = z.index_select(-1, tt.arange(self.indices_unpack['alpha'][0], self.indices_unpack['alpha'][1]))
		beta_prec = z.index_select(-1, tt.arange(self.indices_unpack['beta'][0], self.indices_unpack['beta'][1]))
		mu = z.index_select(-1, tt.arange(self.indices_unpack['mu'][0], self.indices_unpack['mu'][1]))
		lnsigma_alpha = z.index_select(-1, tt.arange(self.indices_unpack['lns_a'][0], self.indices_unpack['lns_a'][1]))
		lnsigma_beta = z.index_select(-1, tt.arange(self.indices_unpack['lns_b'][0], self.indices_unpack['lns_b'][1]))
		# print(alpha_eth.shape, beta_prec.shape, mu.shape, lnsigma_alpha.shape, lnsigma_beta.shape)
		return alpha_eth, beta_prec, mu, lnsigma_alpha, lnsigma_beta

	def log_prob(self, z):
		alpha, beta, mu, lnsa, lnsb = self._unpack(z)
		return self.logprior(alpha, beta, mu, lnsa, lnsb) + self.loglikelihood(alpha, beta, mu, lnsa, lnsb)

	def logprior_z(self, z):
		alpha, beta, mu, lnsa, lnsb = self._unpack(z)
		return self.logprior(alpha, beta, mu, lnsa, lnsb)

	def logprior(self, alpha, beta, mu, lnsa, lnsb):
		dist = tt.distributions.Normal(loc = 0, scale = self.scale_prior)
		prior_mu = dist.log_prob(mu).squeeze(-1)
		prior_lnsa = dist.log_prob(lnsa.exp()).squeeze(-1)
		prior_lnsb = dist.log_prob(lnsb.exp()).squeeze(-1)
		dist2 = tt.distributions.Normal(loc = 0, scale = lnsa.exp())
		prior_alpha = dist2.log_prob(alpha).sum(-1)
		dist3 = tt.distributions.Normal(loc = 0, scale = lnsb.exp())
		prior_beta = dist3.log_prob(beta).sum(-1)
		return prior_mu + prior_lnsa + prior_lnsb + prior_alpha + prior_beta


	def loglikelihood(self, alpha, beta, mu, lnsa, lnsb):
		# 2D
		# print(self.lnep.shape) # (96) = 32 * 3
		# print(mu.shape) # (a, 1)
		# print(alpha.shape) # (a, 2)
		# print(beta.shape) # (a, 32)
		# print(self.Xeth.shape) # (96, 2)
		# print(self.Xprecinct.shape) # (96, 32)
		# print(self.yep.shape) # (96)
		lnep = tt.from_numpy(self.lnep).float()
		Xeth = tt.from_numpy(self.Xeth).float()
		Xprecinct = tt.from_numpy(self.Xprecinct).float()
		yep = tt.from_numpy(self.yep).float()
		aux = mu + lnep.unsqueeze(0) + alpha @ Xeth.t() + beta @ Xprecinct.t() # lnlambda
		dist = tt.distributions.Poisson(aux.exp())
		return dist.log_prob(yep).sum(-1)



class BNNRW(tt.distributions.Distribution):
	def __init__(self):
		self.name = 'BNNRW'
		self.load_data()
		self.isize = self.x.shape[1]
		self.hsize = 50
		self.osize = 1
		self.scale_prior = 10.
		self.dim = self.isize * self.hsize + self.hsize + self.hsize * self.osize + self.osize + 2 # +2 for the variances
		self.indices_unpack = {'W1': [0, self.isize * self.hsize]}
		self.indices_unpack['b1'] = [self.indices_unpack['W1'][1], self.indices_unpack['W1'][1] + self.hsize]
		self.indices_unpack['W2'] = [self.indices_unpack['b1'][1], self.indices_unpack['b1'][1] + self.hsize * self.osize]
		self.indices_unpack['b2'] = [self.indices_unpack['W2'][1], self.indices_unpack['W2'][1] + self.osize]
		self.indices_unpack['lnalpha'] = [self.indices_unpack['b2'][1], self.indices_unpack['b2'][1] + 1]
		self.indices_unpack['lntau'] = [self.indices_unpack['lnalpha'][1], self.indices_unpack['lnalpha'][1] + 1]

	def load_data(self):
		# Data preprocessing and standardization extracted from paper
		# "Reducing reparameterization gradient variance" by Miller et al. (2017)
		data_file = 'datasets/winequality-red.csv'
		data = pd.read_csv(data_file, sep=';')
		X = data.values[:, :-1]
		y = data.values[:,  -1]

		split_seed, test_fraction = 0, 0.1
		rs = np.random.RandomState(split_seed)
		permutation = rs.permutation(X.shape[0])
		size_train  = int(np.round(X.shape[0] * (1 - test_fraction)))
		index_train = permutation[0:size_train]
		index_test  = permutation[size_train:]

		X_train = X[index_train, :]
		y_train = y[index_train]
		X_test  = X[index_test, :]
		y_test  = y[index_test]
		self.x = torch.from_numpy(X_train[:100, :]).float()
		self.y = torch.from_numpy(y_train[:100]).float()
		self.make_standardize_funs()
		self.standardize_data()

	def make_standardize_funs(self):
		Xtrain = self.x
		Ytrain = self.y
		N = Ytrain.shape[0]
		self.std_Xtrain = Xtrain.std(0) * np.sqrt((N - 1.) / N)
		self.std_Xtrain[self.std_Xtrain == 0] = 1.
		self.mean_Xtrain = Xtrain.mean(0)

		self.std_Ytrain = Ytrain.std(0) * np.sqrt((N - 1.) / N)
		self.mean_Ytrain = Ytrain.mean(0)

		self.std_X = lambda X: (X - self.mean_Xtrain) / self.std_Xtrain
		self.ustd_X = lambda X: X * self.std_Xtrain + self.mean_Xtrain

		self.std_Y = lambda Y: (Y - self.mean_Ytrain) / self.std_Ytrain
		self.ustd_Y = lambda Y: Y * self.std_Ytrain + self.mean_Ytrain

	def standardize_data(self):
		self.x = self.std_X(self.x)
		self.y = self.std_Y(self.y)

	def _unpack(self, z):
		W1 = z.index_select(-1, tt.arange(self.indices_unpack['W1'][0], self.indices_unpack['W1'][1])).view(-1, self.isize, self.hsize)
		b1 = z.index_select(-1, tt.arange(self.indices_unpack['b1'][0], self.indices_unpack['b1'][1])).unsqueeze(-2)
		W2 = z.index_select(-1, tt.arange(self.indices_unpack['W2'][0], self.indices_unpack['W2'][1])).view(-1, self.hsize, self.osize)
		b2 = z.index_select(-1, tt.arange(self.indices_unpack['b2'][0], self.indices_unpack['b2'][1])).unsqueeze(-2)
		lnalpha = z.index_select(-1, tt.arange(self.indices_unpack['lnalpha'][0], self.indices_unpack['lnalpha'][1]))
		lntau = z.index_select(-1, tt.arange(self.indices_unpack['lntau'][0], self.indices_unpack['lntau'][1]))
		# print(W1.shape, W2.shape, b1.shape, b2.shape, lnalpha.shape, lntau.shape)
		# (a, 11, 50) (a, 50, 1) (a, 1, 50) (a, 1, 1)
		return W1, b1, W2, b2, lnalpha, lntau

	def _feed_forward(self, x, W1, b1, W2, b2):
		aux = tt.matmul(x, W1) # (a, N, 50)
		aux = aux + b1 # (a, N, 50)
		aux = F.relu(aux) # (a, N, 50)
		aux = tt.matmul(aux, W2) # (a, N, 1)
		aux = aux + b2 # (a, N, 1)
		return aux.squeeze(-1) # (a, N)

	def log_prob(self, z):
		W1, b1, W2, b2, lnalpha, lntau = self._unpack(z)
		return self.logprior(W1, b1, W2, b2, lnalpha, lntau) + self.loglikelihood(W1, b1, W2, b2, lnalpha, lntau)

	def logprior_z(self, z):
		W1, b1, W2, b2, lnalpha, lntau = self._unpack(z)
		return self.logprior(W1, b1, W2, b2, lnalpha, lntau)

	def logprior(self, W1, b1, W2, b2, lnalpha, lntau):
		a0, b0 = 1, 0.1
		prior_lnalpha = ((a0 - 1) * lnalpha - b0 * lnalpha.exp() + lnalpha).squeeze(-1)
		prior_lntau = ((a0 - 1) * lntau - b0 * lntau.exp() + lntau).squeeze(-1)

		lnalpha = lnalpha.unsqueeze(-1)
		dist = tt.distributions.Normal(loc = 0, scale = (-lnalpha / 2.).exp())
		prior_W1 = dist.log_prob(W1).sum(-1).sum(-1)
		prior_b1 = dist.log_prob(b1).sum(-1).sum(-1)
		prior_W2 = dist.log_prob(W2).sum(-1).sum(-1)
		prior_b2 = dist.log_prob(b2).sum(-1).sum(-1)
		logprior = prior_W1 + prior_b1 + prior_W2 + prior_b2
		return prior_lnalpha + prior_lntau + logprior

	def loglikelihood(self, W1, b1, W2, b2, lnalpha, lntau):
		y_out = self._feed_forward(self.x, W1, b1, W2, b2).squeeze(-1) # (a, N), self.y is (N)
		dist = tt.distributions.Normal(loc = y_out, scale = (-lntau / 2.).exp())
		lp = dist.log_prob(self.y)
		return lp.sum(-1)

	def set_data(self, x, y):
		self.x = x
		self.y = y
