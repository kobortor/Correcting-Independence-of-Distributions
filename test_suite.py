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
        shifted_dist=shifted_dist/shifted_dist.sum()
        # TODO: Fails
        assert(Distribution2D(shifted_dist).dist_independent() < eps)
#Test if the algorithm works
def test_origin_dist_3(n: int, t: int):
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
        shifted_dist=shifted_dist/shifted_dist.sum()

        #Sample from shifted_dist, then produce a corrected distribution
        P_dis=Distribution2D(shifted_dist)
        #1.Sample from shifted_dist
        #r is sample ratio
        r=int(pow(n,1.8))
        X_t=np.zeros((1,n))
        Y_t=np.zeros((1,n))
        sampled_set=P_dis.sample(r)
        # print(sampled_set)
        # print(sampled_set.shape)
        for sampled in range(r):
            X_t[0,int(sampled_set[sampled,0])]+=1
            Y_t[0,int(sampled_set[sampled,1])]+=1
        X_t=X_t/X_t.sum()
        Y_t=Y_t/Y_t.sum()
        cum_X_t=X_t.cumsum()
        cum_Y_t = Y_t.cumsum()

        #2. Produce a corrected distribution t
        t=np.zeros((n,n))
        m=pow(n,2)
        idx = np.searchsorted(cum_X_t, np.random.random(m))
        idy = np.searchsorted(cum_Y_t, np.random.random(m))
        for new in range(m):
            # cur_x=idx[new]
            # cur_y = int(Y_t[0, np.random.randint(r)])
            t[idx[new],idy[new]]+=1
        #Normalize t
        t=t/t.sum()
        t=Distribution2D(t)
        #3.Test the distance between t and original distribution
        print(P_dis.total_variation(t))
        assert(P_dis.total_variation(t) < 1000*eps)
