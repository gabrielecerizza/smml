import numpy as np
from numpy import linalg as la
from abc import ABCMeta, abstractmethod


class Kernel(metaclass=ABCMeta):
    def __call__(self, *args):
        return self.compute(*args)

    @abstractmethod
    def compute(self, x1, x2):
        pass

    @abstractmethod
    def compute_kernel_matrix(self, X):
        pass

class GaussianKernel(Kernel):
    def __init__(self, gamma=0.25):
        self.gamma = gamma

    def compute(self, x1, x2):
        return np.exp(-la.norm(x1 - x2) ** 2 / (2 * self.gamma))
    
    def compute_kernel_matrix(self, X):
        # TODO: explain newaxis
        return np.exp(
            -(np.linalg.norm(X[:,np.newaxis,:] - X, axis=-1) ** 2) 
            / (2 * self.gamma))
    
