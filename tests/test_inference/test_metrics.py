import numpy as np
import numpy.testing as npt

def test_alpha_correlation():
    from osl_dynamics.inference.metrics import alpha_correlation

    alpha_1 = [np.array([[1, 0, -1], [1, -1, 0]]).T]
    alpha_2 = [np.array([[1, 0, -1], [-1, 1, 0]]).T]

    npt.assert_almost_equal(alpha_correlation(alpha_1, alpha_2,return_diagonal=False),
                            np.array([[1., -0.5], [0.5, -1]]))
    npt.assert_almost_equal(alpha_correlation(alpha_1, alpha_2, return_diagonal=True),
                            np.array([1., -1.]))
