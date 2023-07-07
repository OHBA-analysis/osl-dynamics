import numpy as np
import numpy.testing as npt

def test_z_score():
    
    # Example 1: one session
    x = np.array([[0.5,1.5,2.5],[-2.0,-1.0,0.0]]).T
    y = z_score(x)
    npt.assert_equal(y[:,0], y[:,1])
    
    # Example 2: two sessions
    np.random.seed(42)
    x1 = np.random.normal(loc=100.0,scale=100.0,size=(100,2))
    x2 = np.random.normal(loc=0.0,scale=2.0,size=(100,2))
    x = np.concatenate([x1,x2])
    y = z_score(x,n_session=2)
    
    # Check the data are z-scored
    npt.assert_almost_equal(np.mean(y[:100,:]),0.0,decimal=3)
    npt.assert_almost_equal(np.std(y[:100,:]),1.0,decimal=3)
    npt.assert_almost_equal(np.mean(y[100:,:]),0.0,decimal=3)
    npt.assert_almost_equal(np.std(y[100:,:]),1.0,decimal=3)
    