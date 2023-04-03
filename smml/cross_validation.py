import time
from collections import defaultdict
from itertools import product

import numpy as np
from tqdm.auto import tqdm


class KFoldCrossValidation:
    def __init__(self, n_folds=5, seed=42):
        self.n_folds = n_folds
        self.seed = seed
    
    def cross_validate(self, estimator, X, y):
        self._y_tests, times, errors, accuracies = [], [], [], []

        for X_train, X_test, y_train, y_test in tqdm(
                self._generate_folds(X, y),
                desc='Running CV',
                total=self.n_folds,
                leave=False):
            start = time.process_time()
            estimator.fit(X_train, y_train)
            score = estimator.score(X_test, y_test)
            times.append(time.process_time() - start)
            errors.append(score['zero_one_loss'])
            accuracies.append(score['accuracy'])
            self._y_tests.append(y_test)

        return {
            'error': np.mean(errors),
            'accuracy': np.mean(accuracies),
            'time': np.mean(times)
        }

    def _generate_folds(self, X, y):
        rng = np.random.default_rng(self.seed)
        folds = defaultdict(lambda: [])
        classes, y_indices, class_counts = np.unique(
            y, return_inverse=True, return_counts=True)
        class_indices = np.split(
            np.argsort(y_indices), np.cumsum(class_counts)[:-1])

        for i in range(classes.shape[0]):
            class_indices[i] = rng.permutation(class_indices[i])
            cumsum_counts = [
                int(j * (1 / self.n_folds) * class_indices[i].shape[0]) 
                for j in range(1, self.n_folds)
            ]
            for fold_num, fold_data_indices in enumerate(
                np.split(class_indices[i], cumsum_counts)):
                folds[fold_num].extend(fold_data_indices)

        folds = {i: rng.permutation(folds[i]) 
                 for i in range(self.n_folds)}

        for i in range(self.n_folds):
            yield (np.delete(X, folds[i], axis=0), 
                   np.take(X, folds[i], axis=0), 
                   np.delete(y, folds[i], axis=0), 
                   np.take(y, folds[i], axis=0))     
    
    def plot_folds_stratification(self):
        # TODO: np.bincount(np.take(y, folds[0])), 
        # np.bincount(np.take(y, folds[1]))
        pass


class ParamGridCrossValidation:
    def __init__(self, estimator, param_grid, cv):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv

    def _generate_params(self):
        param_tuples = [
            list(product(*item)) 
            for item in self.param_grid.items()
        ]
        return [dict(el) for el in product(*param_tuples)]

    def fit(self, X, y):
        cv_results = dict()
        for params in tqdm(
            self._generate_params(), 
            desc='Looping on parameters grid'):
            self.estimator.set_params(params)
            cv_result = self.cv.cross_validate(self.estimator, X, y)
            cv_results[tuple(params.items())] = cv_result

        return cv_results