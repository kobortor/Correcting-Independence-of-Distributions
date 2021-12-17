import numpy as np

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
        self.__n = arr.shape[0]

        self.__weights = arr.flatten()
        self.__cum_weights = self.__weights.cumsum()

    # Simple getter
    def n(self) -> int:
        return self.__n

    # Returns (m x 2)-matrix
    # [[r1 c1]
    #  [r2 c2]
    #     .
    #     .
    #     .
    #  [rm cm]]
    def sample(self, m: int=1) -> np.array:
        assert(m > 0)
        self.__samples_used += m
        idx = np.searchsorted(self.__cum_weights, np.random.random(m))
        return np.vstack([idx // self.__n, idx % self.__n]).T

    # Does this work?
    def nearest_independent(self) -> 'Distirbution2D':
        tmp_arr = self.__weights.reshape((self.__n, self.__n))
        p1 = tmp_arr.sum(axis=1).reshape((self.__n, 1))
        p2 = tmp_arr.sum(axis=0).reshape((1, self.__n))

        return Distribution2D(p1.dot(p2))

    # Gets the distance to the nearest independent (product) distrbution 
    def dist_independent(self) -> float:
        return self.total_variation(self.nearest_independent())

    # Gets the total variation distance from the matrix 
    def total_variation(self, other: 'Distribution2D') -> float:
        return np.abs(self.__weights - other.__weights).sum() / 2
