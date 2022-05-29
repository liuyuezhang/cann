import numpy as np


class RandomAgent:
    def __init__(self, batch_size, mu=0.13*2*np.pi, sigma=5.76*2, mu_high=None, sigma_high=None):
        self.batch_size = int(batch_size)
        self.mu_low = mu  # forward velocity rayleigh dist scale (m/sec)
        self.sigma_low = sigma  # std rotation velocity (rads/sec)
        self.mu_high = mu_high if mu_high is not None else mu
        self.sigma_high = sigma_high if sigma_high is not None else sigma
        self.mu = None
        self.sigma = None
        self.reset()

    def act(self):
        v = np.random.rayleigh(self.mu, self.batch_size).astype(np.float32)
        omega = np.random.normal(0., self.sigma, self.batch_size).astype(np.float32)
        return v, omega

    def reset(self):
        self.mu = np.random.uniform(low=self.mu_low, high=self.mu_high)
        self.sigma = np.random.uniform(low=self.sigma_low, high=self.sigma_high)