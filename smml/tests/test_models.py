import numpy as np

from smml.kernels import GaussianKernel


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
            s = (alphas2 * Y).dot(K[i])
            if (Y[i] / (l * t)) * s < 1:
                alphas2[i] += 1

        assert np.all(alphas1 == alphas2)


def test_pegasos_predict():
    N_train = 100
    N_test = 50
    
    for _ in range(100):
        alphas = np.random.rand(N_train)
        y_train = np.random.randint(2, size=N_train)
        X_train = np.random.rand(N_train, 10)
        kernel = GaussianKernel()
        X = np.random.rand(N_test, 10)

        res1 = [
            np.sum([alphas[j] * y_train[j] * kernel(X_train[j], x) 
                    for j in range(X_train.shape[0])]) 
            for x in X
        ]

        K = kernel.compute_kernel_matrix(X, X_train)
        res2 = (alphas * y_train).dot(K)

        np.allclose(res1, res2)
