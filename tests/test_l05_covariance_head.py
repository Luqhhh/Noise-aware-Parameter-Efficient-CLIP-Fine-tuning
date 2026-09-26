import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.covariance import oas

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from l05_covariance_support import fit_discriminants


def fixture():
    rng = np.random.default_rng(42)
    y = np.arange(150) % 3
    x = rng.normal(size=(150, 5)) @ rng.normal(size=(5, 5)) + y[:, None]
    return x, y


def test_oas_matches_independent_sklearn():
    x, y = fixture()
    heads, means, info = fit_discriminants(x, y, 3)
    covariance, shrinkage = oas(x - means[y], assume_centered=True)
    np.testing.assert_allclose(heads['CH01']['covariance'], covariance, rtol=1e-12)
    assert info['oas_shrinkage'] == pytest.approx(shrinkage, abs=1e-12)


def test_linear_head_equals_direct_gaussian_distances():
    x, y = fixture()
    heads, means, _ = fit_discriminants(x, y, 3)
    for head in heads.values():
        precision = np.linalg.inv(head['covariance'])
        difference = x[:, None, :] - means[None, :, :]
        direct = -.5 * np.einsum('nci,ij,ncj->nc', difference, precision, difference)
        linear = x @ head['weight'].T + head['bias']
        np.testing.assert_allclose(linear - linear[:, :1], direct - direct[:, :1], atol=1e-11)


def test_class_permutation_equivariance():
    x, y = fixture()
    permutation = np.array([2, 0, 1])
    original, _, _ = fit_discriminants(x, y, 3)
    shuffled, _, _ = fit_discriminants(x, permutation[y], 3)
    for arm in original:
        np.testing.assert_allclose(original[arm]['weight'], shuffled[arm]['weight'][permutation])


def test_missing_class_rejected():
    x, y = fixture()
    with pytest.raises(ValueError, match='every class'):
        fit_discriminants(x, y, 4)
