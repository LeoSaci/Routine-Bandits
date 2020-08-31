import numpy as np


class GaussianBandit(object):
    def __init__(self, mu, sigma=1):
        self.mu = mu
        self.gaps = np.max(mu) - mu
        self.sigma = sigma
        self.K = mu.shape[0]
        self.cov = sigma * np.ones(self.K)

    def pull(self, i):
        return np.random.normal(loc=self.mu[i], scale=self.sigma)

    def sample(self, N):
        return np.random.normal(loc=self.mu, scale=self.cov, size=(N, self.K))