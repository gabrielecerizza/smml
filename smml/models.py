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

    def fit(self, X, y, K=None):
        if (self.l <= 0): 
            raise ValueError(
                'Parameter l (lambda) is not strictly positive')
        
        self.y_train_ = y
        self.alphas_ = np.zeros(X.shape[0])
        rng = np.random.default_rng(self.seed)
        if K is None:
            K = self.kernel.compute_kernel_matrix(X, X)
            self.X_train_ = X

        for t in range(1, self.T + 1):
            i = rng.integers(X.shape[0])
            s = self.alphas_.dot(y * K[i])
            if (y[i] / (self.l * t)) * s < 1:
                self.alphas_[i] += 1

    def predict(self, X=None, K=None):
        #TODO: check if we can remove eta, sign should not change
        # eta positive, so monotone function, does not change order
        if K is None:
            K = self.kernel.compute_kernel_matrix(X, self.X_train_)

        return (self.alphas_ * self.y_train_).dot(K)


class MulticlassPegasos:
    def __init__(
            self, l=0.5, T=1000, kernel=GaussianKernel(), seed=42):
        self.l = l
        self.T = T
        self.kernel = kernel
        self.seed = seed
        
    def fit(self, X, y):
        if (self.l <= 0): 
            raise ValueError(
                'Parameter l (lambda) is not strictly positive')
        
        self.X_train_ = X
        self.y_train_ = y
        self.predictors_ = []
        K = self.kernel.compute_kernel_matrix(X, X)

        for class_label in tqdm(
            np.unique(y), desc='Training predictors'):
            p = Pegasos(self.l, self.T, self.kernel, self.seed)
            y_enc = np.where(y == class_label, 1, -1)
            p.fit(X, y_enc, K)
            self.predictors_.append(p)

    def score(self, X, y):
        K = self.kernel.compute_kernel_matrix(X, self.X_train_)
        preds = np.array(
            [p.predict(K=K) for p in self.predictors_])
        y_pred = np.argmax(preds.T, axis=-1)
        accuracy = np.mean(y == y_pred)

        return {
            'zero_one_loss': 1 - accuracy,
            'accuracy': accuracy
        }
