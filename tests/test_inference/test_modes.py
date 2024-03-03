import numpy as np
import numpy.testing as npt


def test_argmax_time_courses():
    from osl_dynamics.inference.modes import argmax_time_courses

    # Test case 1: When alpha is a list, concatenate is false
    alpha_list = [np.array([[0.1, 0.9], [0.8, 0.2]]), np.array([[0.2, 0.8], [0.6, 0.4]])]
    expected_result = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
    result = argmax_time_courses(alpha_list, concatenate=False)
    npt.assert_array_equal(result, expected_result)

    # Test case 2: When alpha is a list, concatenate is true
    expected_result = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    result = argmax_time_courses(alpha_list, concatenate=True)
    npt.assert_array_equal(result, expected_result)

    # Test case 3: When alpha is a 2D numpy array
    alpha_2d = np.array([[0.1, 0.9], [0.8, 0.2]])
    expected_result = np.array([[0, 1], [1, 0]])
    result = argmax_time_courses(alpha_2d)
    npt.assert_array_equal(result, expected_result)

    # Test case 4: When alpha is a 3D numpy array
    alpha_3d = np.array([[[0.1, 0.9], [0.8, 0.2]], [[0.2, 0.8], [0.6, 0.4]]])
    expected_result = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
    result = argmax_time_courses(alpha_3d)
    npt.assert_array_equal(result, expected_result)


def test_hungarian_pair():
    from osl_dynamics.inference.modes import hungarian_pair

    # When distance is true
    matrix = np.array([[8, 4, 7], [5, 2, 3], [9, 4, 8]])
    indices, matrix_reordered = hungarian_pair(matrix, distance=True)
    matrix_reordered_true = np.array([[8, 7, 4], [5, 3, 2], [9, 8, 4]])
    npt.assert_equal(np.array(indices['row']), [0, 1, 2])
    npt.assert_equal(np.array(indices['col']), [0, 2, 1])
    npt.assert_equal(matrix_reordered, matrix_reordered_true)

    # When distance is False
    indices, matrix_reordered = hungarian_pair(-matrix, distance=False)
    matrix_reordered_true = - np.array([[8, 7, 4], [5, 3, 2], [9, 8, 4]])
    npt.assert_equal(np.array(indices['row']), [0, 1, 2])
    npt.assert_equal(np.array(indices['col']), [0, 2, 1])
    npt.assert_equal(matrix_reordered, matrix_reordered_true)


def test_reduce_state_time_course():
    from osl_dynamics.inference.modes import reduce_state_time_course

    state_time_course = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 0],
        [1, 1, 1]
    ]).T
    expected_result = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 1]
    ]).T
    reduced_state_time_course = reduce_state_time_course(state_time_course)
    npt.assert_equal(reduced_state_time_course, expected_result)


def test_correlate_modes():
    from osl_dynamics.inference.modes import correlate_modes

    alpha_1 = np.array([[1, 0, -1], [1, -1, 0]]).T
    alpha_2 = np.array([[1, 0, -1], [-1, 1, 0]]).T

    npt.assert_almost_equal(correlate_modes(alpha_1, alpha_2),
                            np.array([[1., -0.5], [0.5, -1]]))
