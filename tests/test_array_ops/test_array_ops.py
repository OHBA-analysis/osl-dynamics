import numpy as np
import numpy.testing as npt

def test_get_one_hot():
    from osl_dynamics.array_ops import get_one_hot

    # Case 1: Categorical input
    input_1 = np.array([0,2,0,1])
    output_1 = np.array([
        [1,0,0],
        [0,0,1],
        [1,0,0],
        [0,1,0]
    ])
    npt.assert_equal(get_one_hot(input_1),output_1)

    # Case 2: Categorical input, but input n_states
    input_2 = np.array([0, 2, 0, 1])
    output_2 = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    npt.assert_equal(get_one_hot(input_2,n_states=4), output_2)

    # Case 3: (n_samples, n_states) to be binarized
    input_3 = np.array([
        [0.99,0.03,0.03],
        [0.02,0.02,0.9],
        [0.80,0.4,0.5],
        [-1.,-0.5,-1.]
    ])
    output_3 = np.array([
        [1,0,0],
        [0,0,1],
        [1,0,0],
        [0,1,0]
    ])
    npt.assert_equal(get_one_hot(input_3),output_3)

    # Case 4: (n_samples, n_states) to be binarized, input n_states
    input_4 = np.array([
        [0.99, 0.03, 0.03],
        [0.02, 0.02, 0.9],
        [0.80, 0.4, 0.5],
        [-1., -0.5, -1.]
    ])
    output_4 = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    npt.assert_equal(get_one_hot(input_4,n_states=4), output_4)

def test_cov2std():
    from osl_dynamics.array_ops import cov2std

    # Case 1: One covariance matrix
    cov = np.array([[16.0, 4.0], [4.0, 4.0]])
    std = cov2std(cov)
    npt.assert_equal(std, np.array([4.0, 2.0]))

    # Case 2: Two covariance matrices
    covs = np.array([[[16.0, 4.0], [4.0, 4.0]], [[100.0, -40.0], [-40.0, 400.0]]])
    stds = cov2std(covs)
    npt.assert_equal(stds, np.array([[4.0, 2.0], [10.0, 20.0]]))

def test_cov2corr():
    from osl_dynamics.array_ops import cov2corr

    # Case 1: One covariance matrix
    cov = np.array([[100.0, -40.0], [-40.0, 400.0]])
    corr = cov2corr(cov)
    npt.assert_equal(corr, np.array([[1.0, -0.2], [-0.2, 1.0]]))

    # Case 2: Two covariance matrices
    covs = np.array([[[16.0, 4.0], [4.0, 4.0]], [[100.0, -40.0], [-40.0, 400.0]]])
    corrs = cov2corr(covs)
    npt.assert_equal(corrs, np.array([[[1.0, 0.5], [0.5, 1.0]], [[1.0, -0.2], [-0.2, 1.0]]]))

def test_stdcor2cov():
    from osl_dynamics.array_ops import stdcor2cov
    # Case 1: One covariance matrix, std is a vector
    std = np.array([4.0, 2.0])
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    cov = stdcor2cov(std,corr)
    npt.assert_equal(cov,np.array([[16.0, 4.0], [4.0, 4.0]]))

    # Case 2: Two covariance matrices, std is two vectors
    stds = np.array([[4.0, 2.0], [10.0, 20.0]])
    corrs = np.array([[[1.0, 0.5], [0.5, 1.0]], [[1.0, -0.2], [-0.2, 1.0]]])
    covs = stdcor2cov(stds,corrs)
    npt.assert_equal(covs,np.array([[[16.0, 4.0], [4.0, 4.0]], [[100.0, -40.0], [-40.0, 400.0]]]))

    # Case 3: One covariance matrix, std is a diagonal matrix
    std = np.array([[4.0,0.0],[0.0, 2.0]])
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    cov = stdcor2cov(std, corr,std_diagonal=True)
    npt.assert_equal(cov, np.array([[16.0, 4.0], [4.0, 4.0]]))

    # Case 4: Two covariance matrices, std is two diagonal matrices
    stds = np.array([[[4.0,0.0],[0.0,2.0]], [[10.0,0.0],[0.0,20.0]]])
    corrs = np.array([[[1.0, 0.5], [0.5, 1.0]], [[1.0, -0.2], [-0.2, 1.0]]])
    covs = stdcor2cov(stds, corrs,std_diagonal=True)
    npt.assert_equal(covs, np.array([[[16.0, 4.0], [4.0, 4.0]], [[100.0, -40.0], [-40.0, 400.0]]]))

def test_cov2stdcorr():
    from osl_dynamics.array_ops import cov2stdcorr
    # Case 1: One covariance matrix
    cov = np.array([[16.0, 4.0], [4.0, 4.0]])
    std,corr = cov2stdcorr(cov)
    npt.assert_equal(std, np.array([4.0, 2.0]))
    npt.assert_equal(corr,np.array([[1.0, 0.5], [0.5, 1.0]]))


    # Case 2: Two covariance matrices
    covs = np.array([[[16.0, 4.0], [4.0, 4.0]], [[100.0, -40.0], [-40.0, 400.0]]])
    stds,corrs = cov2stdcorr(covs)
    npt.assert_equal(stds, np.array([[4.0, 2.0], [10.0, 20.0]]))
    npt.assert_equal(corrs, np.array([[[1.0, 0.5], [0.5, 1.0]], [[1.0, -0.2], [-0.2, 1.0]]]))
