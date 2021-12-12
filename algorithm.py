import abc
import numpy as np
from distribution2d import Distribution2D

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
