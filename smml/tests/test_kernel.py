import numpy as np
from numpy import linalg as la

from smml.kernel import GaussianKernel


def test_gaussian_kernel():
    def k(a, b):
        gamma = 0.25
        return np.exp((-1 / (2 * gamma)) * (la.norm(a - b) ** 2))
    
    K = GaussianKernel()

    for i in range(10000):
        x1, x2 = np.random.rand(2,10)
        v1 = K(x1,x2)
        v2 = k(x1,x2)
        assert  v1 == v2, (i, v1, v2)