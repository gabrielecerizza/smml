import numpy as np
from numpy import linalg as la

from smml.kernels import GaussianKernel


def test_gaussian_kernel():
    def k(a, b):
        gamma = 0.25
        return np.exp((-1 / (2 * gamma)) * (la.norm(a - b) ** 2))
    
    K = GaussianKernel()

    for i in range(10000):
        x1, x2 = np.random.rand(2,10)
        v1 = K(x1,x2)
        v2 = k(x1,x2)
        assert v1 == v2, (i, v1, v2)

def test_gaussian_kernel_matrix():
    gamma = 0.25
    
    for i in range(10000):
        X = np.random.rand(100,10)
        res1 = np.zeros((X.shape[0],X.shape[0]))
        K = GaussianKernel()
        for i in range(0,X.shape[0]):
            for j in range(0,X.shape[0]):
                res1[i,j] = K(X[i], X[j])

        res2 = np.exp(
            -(np.linalg.norm(X[:,np.newaxis,:] - X, axis=-1) ** 2) 
            / (2 * gamma))
        
        assert np.all(res1 == res2)

def test_pegasos_fit():
    l = 0.5

    for i in range(100): 
        N = 100
        X = np.random.rand(N, 10)
        Y = np.random.randint(2, size=N)
        K = np.random.rand(N, N)

        alphas1 = np.zeros(X.shape[0])
        rng = np.random.default_rng(42)
        for t in range(1, 100):
            i = rng.integers(X.shape[0])
            s = np.sum(
                [alphas1[j] * Y[j] * K[i,j] 
                    for j in range(X.shape[0])])
            if (Y[i] / (l * t)) * s < 1:
                alphas1[i] += 1

        alphas2 = np.zeros(X.shape[0])
        rng = np.random.default_rng(42)
        for t in range(1, 100):
            i = rng.integers(X.shape[0])
            s = alphas2.dot(Y * K[i])
            if (Y[i] / (l * t)) * s < 1:
                alphas2[i] += 1

        assert np.all(alphas1 == alphas2)