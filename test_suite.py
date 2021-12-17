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