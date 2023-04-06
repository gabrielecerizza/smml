from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import linalg as la
from tqdm import tqdm


class Kernel(metaclass=ABCMeta):
    def __call__(self, *args):
        return self.compute(*args)

    @abstractmethod
    def compute(self, x1, x2):
        pass

    @abstractmethod
    def compute_kernel_matrix(self, X):
        pass

    def compute_kernel_matrix_slow(self, X1, X2):
        res = np.zeros((X1.shape[0], X2.shape[0]))
        for i in tqdm(range(0, X1.shape[0])):
            for j in range(0, X2.shape[0]):
                res[i,j] = self.compute(X1[i], X2[j])
        return res


class GaussianKernel(Kernel):
    def __init__(self, gamma=0.25):
        self.gamma = gamma

    def compute(self, x1, x2):
        return np.exp(-la.norm(x1 - x2) ** 2 / (2 * self.gamma))
    
    def compute_kernel_matrix(self, X1, X2):
        X2_splits = np.array_split(X2, 20)
        res = []
        for split in X2_splits:
            res.append(self.compute_kernel_matrix_fast(split, X1))
        return np.vstack(res)
    
    def compute_kernel_matrix_fast(self, X1, X2):
        # TODO: explain newaxis
        return np.exp(
            -(np.linalg.norm(X1[:,np.newaxis,:] - X2, axis=-1) ** 2) 
            / (2 * self.gamma))
    

class PolynomialKernel(Kernel):
    def __init__(self, n=3):
        self.n = n

    def compute(self, x1, x2):
        return (1 + x1.dot(x2)) ** self.n
    
    def compute_kernel_matrix(self, X1, X2):
        return (1 + X2.dot(X1.T)) ** self.n

    
    
