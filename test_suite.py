import numpy as np
from distribution2d import Distribution2D

# Make sure the independence test does not fail the normal test
def test_independent_1(n: int, t: int):
    for i in range(t):
        px = np.random.random((n, 1))
        py = np.random.random((1, n))

        px /= px.sum()
        py /= py.sum()

        assert(Distribution2D(px.dot(py)).dist_independent() < 1e-5)

# Test that the distances are accurate
def test_independent_2(n: int, t: int):
    for i in range(t):
        # Randomly generates an epsilon between 10^-6 and 10^-2
        eps = np.power(10, np.random.uniform(low=-6, high=-2))

        px = np.random.random((n, 1))
        py = np.random.random((1, n))

        px /= px.sum()
        py /= py.sum()

        tmp_dist = px.dot(py)

        # Generate a distribution with absolute distance at most half eps distance away
        delta = np.random.randn(n, n)
        delta = (0.5 * eps) * delta / np.abs(delta).sum()
        shifted_dist = tmp_dist + delta

        # Clip everything so we don't have negative values or anything 
        low_bound = np.minimum(tmp_dist, np.full((n, n), 1e-6))
        high_bound = np.maximum(tmp_dist, np.full((n, n), 1-1e-6))
        shifted_dist = np.clip(shifted_dist, low_bound, high_bound)

        # TODO: Fails
        assert(Distribution2D(shifted_dist).dist_independent() < eps)