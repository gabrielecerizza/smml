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
        K = self.kernel.compute_kernel_matrix(X)

        for t in tqdm(range(1, self.T + 1)):
            i = rng.integers(X.shape[0])
            s = self.alphas.dot(y * K[i])
            if (y[i] / (self.l * t)) * s < 1:
                self.alphas[i] += 1

    def predict(self, X):
        #TODO: check if we can remove eta, sign should not change

        return np.array(
            [np.sign((1 / (self.l * self.T)) 
                     * np.sum([self.alphas[j] * self.y_train[j] * self.K(self.X_train[j], x) 
                               for j in range(self.X_train.shape[0])])) for x in X])