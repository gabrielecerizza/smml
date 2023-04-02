import numpy as np
from tqdm.auto import tqdm

from smml.kernels import GaussianKernel


class Pegasos:
    def __init__(
            self, l=0.5, T=1000, kernel=GaussianKernel(), seed=42):
        self.l = l
        self.T = T
        self.kernel = kernel
        self.seed = seed

    def fit(self, X : np.ndarray, y : np.ndarray):
        if (self.l <= 0): 
            raise ValueError(
                'Parameter l (lambda) is not strictly positive')

        self.X_train = X
        self.y_train = y
        self.alphas = np.zeros(X.shape[0])
        rng = np.random.default_rng(self.seed)
        K = self.kernel.compute_kernel_matrix(X, X)

        for t in tqdm(range(1, self.T + 1)):
            i = rng.integers(X.shape[0])
            s = self.alphas.dot(y * K[i])
            if (y[i] / (self.l * t)) * s < 1:
                self.alphas[i] += 1

    def predict_proba(self, X):
        #TODO: check if we can remove eta, sign should not change

        K = self.kernel.compute_kernel_matrix(X, self.X_train)
        eta = 1 / (self.l * self.T)
        return eta * ((self.alphas * self.y_train).dot(K))
