"""Training-only pooled Gaussian discriminants, compiled into one linear head."""
import numpy as np


def fit_discriminants(features, labels, classes):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError('Invalid feature matrix or labels')
    if np.any(y < 0) or np.any(y >= classes):
        raise ValueError('Label outside mapping')
    counts = np.bincount(y, minlength=classes)
    if np.any(counts == 0):
        raise ValueError('Training must cover every class')
    means = np.zeros((classes, x.shape[1]), dtype=np.float64)
    np.add.at(means, y, x)
    means /= counts[:, None]
    residual = x - means[y]
    covariance = residual.T @ residual / len(x)
    dimension = x.shape[1]
    mu = np.trace(covariance) / dimension
    if mu <= 0:
        raise ValueError('Zero within-class variance')
    # OAS formula used by sklearn (large-p approximation), assume_centered=True.
    alpha = np.mean(covariance ** 2)
    denominator = (len(x) + 1) * (alpha - mu ** 2 / dimension)
    shrinkage = 1. if denominator <= 0 else min((alpha + mu ** 2) / denominator, 1.)
    shrunk = (1 - shrinkage) * covariance + shrinkage * mu * np.eye(dimension)
    matrices = {'CH00': mu * np.eye(dimension), 'CH01': shrunk}
    heads = {}
    for arm, matrix in matrices.items():
        weight = np.linalg.solve(matrix, means.T).T
        bias = -.5 * (means * weight).sum(axis=1)
        heads[arm] = {'weight': weight, 'bias': bias, 'covariance': matrix}
    return heads, means, {'samples': len(x), 'classes': classes, 'dimension': dimension,
                          'oas_shrinkage': float(shrinkage), 'pooled_variance': float(mu),
                          'min_class_count': int(counts.min()),
                          'max_class_count': int(counts.max())}
