import numpy as np
import numpy.testing as npt


def test_alpha_correlation():
    from osl_dynamics.inference.metrics import alpha_correlation

    alpha_1 = [np.array([[1, 0, -1], [1, -1, 0]]).T]
    alpha_2 = [np.array([[1, 0, -1], [-1, 1, 0]]).T]

    npt.assert_almost_equal(alpha_correlation(alpha_1, alpha_2, return_diagonal=False),
                            np.array([[1., -0.5], [0.5, -1]]))
    npt.assert_almost_equal(alpha_correlation(alpha_1, alpha_2, return_diagonal=True),
                            np.array([1., -1.]))


def test_confusion_matrix():
    from osl_dynamics.inference.metrics import confusion_matrix

    # Define two example state time courses
    state_time_course_1 = np.array([0, 1, 1, 0, 2])  # Example states
    state_time_course_2 = np.array([0, 1, 2, 2, 2])  # Example states

    # Expected confusion matrix
    expected_cm = np.array([[1, 0, 1],
                            [0, 1, 1],
                            [0, 0, 1]])

    # Calculate the confusion matrix
    cm = confusion_matrix(state_time_course_1, state_time_course_2)

    # Check if the calculated confusion matrix matches the expected one
    npt.assert_equal(cm, expected_cm)


def test_dice_coefficient():
    from osl_dynamics.inference.metrics import dice_coefficient
    # Test case 1: One-dimensional input
    sequence_1 = np.array([0, 1, 1, 0, 2])
    sequence_2 = np.array([0, 1, 2, 2, 2])
    expected_dice = 0.6
    npt.assert_equal(dice_coefficient(sequence_1, sequence_2), expected_dice)

    # Test case 2:  two-dimensional input
    sequence_1 = np.array([
        [0.1, 0.2, 0.3],
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.4, 0.5, 0.6]
    ])
    sequence_2 = np.array([
        [0.1, 0.3, 0.4],
        [0.5, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.6, 0.5, 0.4]
    ])
    expected_dice = 0.5
    npt.assert_equal(dice_coefficient(sequence_1, sequence_2), expected_dice)


def test_pairwise_fisher_z_correlations():
    from osl_dynamics.inference.metrics import pairwise_fisher_z_correlations
    x1, x2, x3 = -0.2, 0.5, 0.3
    y1, y2, y3 = 0.7, -0.1, 0.4

    def fisher_z_inverse(x):
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    answer = np.corrcoef(np.array([[x1, x2, x3], [y1, y2, y3]]))

    matrices = np.array([[[0.0, x1, x2], [x1, 0.0, x3], [x2, x3, 0.0]],
                         [[0.0, y1, y2], [y1, 0.0, y3], [y2, y3, 0.0]]])
    matrices = fisher_z_inverse(matrices) + np.eye(3)

    npt.assert_almost_equal(pairwise_fisher_z_correlations(matrices), answer, decimal=6)


def test_twopair_fisher_z_correlations():
    from osl_dynamics.inference.metrics import twopair_fisher_z_correlations
    x1, x2, x3 = -0.2, 0.5, 0.3
    y1, y2, y3 = 0.7, -0.1, 0.4

    def fisher_z_inverse(x):
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    answer = np.corrcoef(np.array([[x1, x2, x3], [y1, y2, y3]]))

    matrices = np.array([[[0.0, x1, x2], [x1, 0.0, x3], [x2, x3, 0.0]],
                         [[0.0, y1, y2], [y1, 0.0, y3], [y2, y3, 0.0]]])
    matrices = fisher_z_inverse(matrices) + np.eye(3)

    npt.assert_almost_equal(twopair_fisher_z_correlations(matrices, matrices), answer, decimal=6)


def test_regularisation():
    from osl_dynamics.inference.metrics import regularisation

    matrix = np.eye(2)
    regularised_matrix = regularisation(matrix, 1e-6)
    npt.assert_equal(regularised_matrix, np.eye(2) * (1 + 1e-6))

    matrices = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    regularised_matrix = regularisation(matrices, 1e-6)
    npt.assert_equal(regularised_matrix, np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]) * (1 + 1e-6))


def test_twopair_vector_correlation():
    from osl_dynamics.inference.metrics import twopair_vector_correlation
    vectors_1 = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    vectors_2 = np.array([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]])
    true_correlation = np.array([[1.0, -1.0, ], [1.0, -1.0]])
    npt.assert_equal(true_correlation, twopair_vector_correlation(vectors_1, vectors_2))

def test_twopair_riemannian_distance():
    from osl_dynamics.inference.metrics import twopair_riemannian_distance
    from math import sqrt
    corr_1 = np.array([[1.0,0.0],[0.0,1.0]])
    corr_2 = np.array([[2.0,0.0],[0.0,1.0]])
    corr_3 = np.array([[1.0,0.0],[0.0,2.0]])
    matrices_1 = np.stack([corr_1,corr_2])
    matrices_2 = np.stack([corr_1,corr_3])
    answer = np.array([[0.0,np.log(2)],[np.log(2),sqrt(2) * np.log(2)]])
    npt.assert_almost_equal(twopair_riemannian_distance(matrices_1,matrices_2),answer,decimal=6)
