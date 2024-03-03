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
    npt.assert_equal(dice_coefficient(sequence_1,sequence_2),expected_dice)

    # Test case 2:  two-dimensional input
    sequence_1 = np.array([
        [0.1, 0.2, 0.3],
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.4,0.5,0.6]
    ])
    sequence_2 = np.array([
        [0.1,0.3,0.4],
        [0.5, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.6,0.5,0.4]
    ])
    expected_dice = 0.5
    npt.assert_equal(dice_coefficient(sequence_1, sequence_2), expected_dice)