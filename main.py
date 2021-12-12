#!/usr/bin/python3
import numpy as np 
import abc

class Distribution2D:
    def __init__(self, arr: np.array):
        # make sure it's a square matrix
        assert arr.ndim == 2
        assert(arr.shape[0] == arr.shape[1])

        # make sure it is a probability distribution
        assert(np.isclose(np.sum(arr), 1))
        assert((arr >= 0).all())

        # save necessary values
        self.__samples_used = 0
        self.__n = arr.shape[0]
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
        return np.vstack([idx // self.__n, idx % self.__n]).T

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

if __name__ == "__main__":
    p = Distribution2D(np.eye(4) / 4)
    # print(p.sample(10))

    algo = NaiveCorrector()
    print(algo.improve(p=p, q=10, eps=0.1, delta=0.2))