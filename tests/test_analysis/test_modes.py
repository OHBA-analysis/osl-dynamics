import numpy as np
import numpy.testing as npt

def test_fractional_occupancies():
    from osl_dynamics.analysis.modes import fractional_occupancies

    # Test case 1: One session with a 2D numpy array
    state_time_course_1 = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    expected_result_1 = np.array([2 / 3, 2 / 3, 1 / 3])
    npt.assert_equal(fractional_occupancies(state_time_course_1), expected_result_1)

    # Test case 2: Multiple sessions with a 3D numpy array
    state_time_course_2 = np.array([[[1, 0, 1], [0, 1, 0], [1, 1, 0]],
                                    [[1, 1, 1], [0, 0, 0], [1, 0, 1]]])
    expected_result_2 = np.array([[2 / 3, 2 / 3, 1 / 3], [2 / 3, 1 / 3, 2 / 3]])
    npt.assert_equal(fractional_occupancies(state_time_course_2), expected_result_2)

    # Test case 3: Multiple sessions with a list of numpy arrays
    state_time_course_3 = [np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]]),
                           np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1]])]
    npt.assert_equal(fractional_occupancies(state_time_course_3), expected_result_2)