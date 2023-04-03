import numpy as np
from numpy import linalg as la

from smml.kernels import GaussianKernel, PolynomialKernel


def test_gaussian_kernel():
    def k(a, b):
        gamma = 0.25
        return np.exp((-1 / (2 * gamma)) * (la.norm(a - b) ** 2))
    
    K = GaussianKernel()

    for i in range(10000):
        x1, x2 = np.random.rand(2, 10)
        v1 = K(x1, x2)
        v2 = k(x1, x2)
        assert v1 == v2, (i, v1, v2)


def test_gaussian_kernel_matrix():
    for i in range(100):
        X = np.random.rand(100, 10)
        res1 = np.zeros((X.shape[0], X.shape[0]))
        K = GaussianKernel()
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[0]):
                res1[i,j] = K(X[i], X[j])

        res2 = K.compute_kernel_matrix(X, X)
        
        assert np.allclose(res1, res2)


def test_gaussian_kernel_matrix_fast():
    for i in range(100):
        X = np.random.rand(100, 10)
        res1 = np.zeros((X.shape[0], X.shape[0]))
        K = GaussianKernel()
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[0]):
                res1[i,j] = K(X[i], X[j])

        res2 = K.compute_kernel_matrix_fast(X, X)
        
        assert np.allclose(res1, res2)


def test_polynomial_kernel_matrix():
    for i in range(100):
        X = np.random.rand(100, 10)
        res1 = np.zeros((X.shape[0], X.shape[0]))
        K = PolynomialKernel()
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[0]):
                res1[i,j] = K(X[i], X[j])

        res2 = K.compute_kernel_matrix(X, X)
        
        assert np.allclose(res1, res2)