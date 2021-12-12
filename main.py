#!/usr/bin/python3
import numpy as np
import abc

# Does this even work?
# TODO: check?
def test_independent(dist: np.array)-> float:
    #For an independent distribution P[x,y]=P[x]*P[y]
    n=dist.shape[0]
    id_dist=np.zeros((n,n))
    probx = np.zeros((1,n))
    proby = np.zeros((1, n))
    for i in range(n):
        probx[0,i] = dist[i,:].sum()
        proby[0,i] = dist[:,i].sum()
    for i in range(n):
        for j in range(n):
            id_dist[i,j]=probx[0,i]*proby[0,j]
    return dTV(dist,id_dist)

def dTV(dist_1: np.array,dist_2: np.array)->float:
    #Here I use the def in Open Problem 69
    #dTV(p,q)=max(p(S)-q(S))
    #Notice the def of closeness in textbook is different
    assert (dist_1.shape == dist_2.shape)
    cur_dist=0
    for i in range(dist_1.shape[0]):
        for j in range(dist_1.shape[1]):
            cur_dist=max(cur_dist,abs(dist_1[i,j]-dist_2[i,j]))
    return cur_dist

class Distribution2D:
    def __init__(self, arr: np.array):
        # make sure it's a square matrix
        assert arr.ndim == 2
        assert(arr.shape[0] == arr.shape[1])

        # make sure it is a probability distribution
        assert(np.isclose(np.sum(arr), 1))
        assert((arr >= 0).all())

        # TODO: check this is close to an independent product distribution?
        # Identity distribution matrices are definite NOT close

        # save necessary values
        self.__samples_used = 0
        self.n = arr.shape[0]
        self.__weights = arr.flatten().cumsum()

    # Returns (m x 2)-matrix
    # [[r1 c1]
    #  [r2 c2]
    #     .
    #     .
    #     .
    #  [rm cm]]
    def sample(self, m=1) -> np.array:
        assert(m > 0)
        self.__samples_used += m
        idx = np.searchsorted(self.__weights, np.random.random(m))
        return np.vstack([idx // self.n, idx % self.n]).T


    def total_variation(self, other) -> float:
        return np.abs(self.__weights - other.__weights).sum() / 2

class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    # Fill in this to improve or correct
    # Given distribution p that is guaranteed to be eps-close to a product distribution
    # Provide, with probability 1-delta, q IID samples of a distribution that is
    #  1) independent
    #  2) O(eps) close to p
    #
    # This is a pure virtual function, you cannot instantiate the `Algorithm` class. Make
    #   sure you inherit it and build your own function!
    @abc.abstractmethod
    def improve(self, p: Distribution2D, q: int, eps: float, delta: float) -> np.ndarray:
        pass

class NaiveCorrector(Algorithm):
    # Sample q things and take their X values, sample q things and take their Y values, join
    def improve(self, p: Distribution2D, q: int, eps: float, delta: float) -> np.ndarray:
        return np.vstack([p.sample(q)[:, 0], p.sample(q)[:, 1]]).T

# Test the algorithm `t` times
def test_1(p: Distribution2D, algo: Algorithm, eps: float, delta: float, t: int):
    n = p.n

    # Just some sub-quadratic number of samples to get
    q = max(100, int(n * np.sqrt(n) + 1))

    sample_results = np.zeros((n, n))

    for i in range(t):
        samples = algo.improve(p, q, eps, delta)
        for x,y in samples:
            sample_results[x,y] += 1

    d_TV_independence = 0. # do we have a good way to test for independence?
    d_TV_distribution = p.total_variation(Distribution2D(sample_results / (t * q)))

    # TODO: find a way to calculate this!
    print(f"Distance to independence: {d_TV_independence}")

    # TODO: make a confidence interval for this!
    print(f"Distance to distribution: {d_TV_distribution}")

if __name__ == "__main__":
    n = 4
    p = Distribution2D(np.full((n, n), 1 / (n * n)))
    algo = NaiveCorrector()
    test_1(p, algo, eps=0.1, delta=0.1, t=100)

    #TODO: Now we know the algorithm is a improver?How to make it a corrector? Or prove it is a corrector?
    # Since the distribution p is e-close to independent, and the "improved" corrector is also close to
    # independent, besides, the expect frequency or probability of P(x,?) P(?,y) in our algorithm is the same
    # as the distribution p. Is is possible to prove it is also e-close?
    #1. I will try to create a random distribution that is e-close to independent first then improve it, and see
    #the distance between improved distribution and original one.
