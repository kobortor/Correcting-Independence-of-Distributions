#!/usr/bin/python3
import numpy as np
import pandas as pd

from distribution2d import Distribution2D
from algorithm import Algorithm, NaiveCorrector
from test_suite import *

# Test the algorithm `t` times
def test_1(p: Distribution2D, algo: Algorithm, eps: float, delta: float, t: int):
    assert p.dist_independent() < eps * 1.01

    n = p.n()

    # Just some sub-quadratic number of samples to get
    q = max(100, int(n * np.sqrt(n) + 1))

    sample_results = np.zeros((n, n))

    rs = []

    for i in range(t):
        p.reset_samples_used()
        samples = algo.improve(p, q, eps, delta)
        rs.append(p.get_samples_used() / q)

        for x,y in samples:
            sample_results[x, y] += 1

    sample_dist = Distribution2D(sample_results / (t * q))
    d_TV_independence = sample_dist.dist_independent() # do we have a good way to test for independence?
    d_TV_distribution = p.total_variation(sample_dist)

    # TODO: find a way to calculate this!
    print("Distance to independence: {:.6f}".format(d_TV_independence))

    # TODO: make a confidence interval for this!
    print("Distance to distribution: {:.6f}".format(d_TV_distribution))

    # Output statistics of rs
    print("### rate statistics ###")
    print(pd.DataFrame(rs).describe())

if __name__ == "__main__":
    test_independent_1(n=100, t=100)
    print("Independence does not have false negatives!")

    test_independent_2(n=100, t=100)
    print("Independence distances are correct!")

    n = 4
    p = Distribution2D(np.full((n, n), 1 / (n * n)))
    algo = NaiveCorrector()
    test_1(p, algo, eps=0.01, delta=0.1, t=100)

    #TODO: Now we know the algorithm is a improver?How to make it a corrector? Or prove it is a corrector?
    # Since the distribution p is e-close to independent, and the "improved" corrector is also close to
    # independent, besides, the expect frequency or probability of P(x,?) P(?,y) in our algorithm is the same
    # as the distribution p. Is is possible to prove it is also e-close?
    #1. I will try to create a random distribution that is e-close to independent first then improve it, and see
    #the distance between improved distribution and original one.
