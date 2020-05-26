from taser import array_ops
import numpy as np
from unittest import TestCase


class TestCorrelateStates(TestCase):
    def setUp(self) -> None:
        self.state_time_course_1 = array_ops.get_one_hot(
            np.random.randint(10, size=1000)
        )
        self.state_time_course_2 = self.state_time_course_1.copy()

    def test_correlate_states(self):

        correlation = array_ops.correlate_states(
            self.state_time_course_1, self.state_time_course_2
        )

        print(correlation)

        self.assertTrue(np.allclose(np.diagonal(correlation), 1))


class TestGetOneHot(TestCase):
    def setUp(self) -> None:
        self.state_time_course = np.random.randint(5, size=100)

    def test_get_one_hot(self):
        one_hot = array_ops.get_one_hot(self.state_time_course)

        self.assertTrue(
            np.all(one_hot.argmax(axis=1) == self.state_time_course),
            msg="One hot did not reproduce the categorical sequence when inverted.",
        )


class TestMatchStates(TestCase):
    def setUp(self) -> None:
        self.state_time_course_1 = array_ops.get_one_hot(
            np.random.randint(10, size=1000)
        )
        self.state_time_course_2 = self.state_time_course_1.copy()

        new_order = np.random.permutation(10)

        self.state_time_course_2 = self.state_time_course_2[:, new_order]

    def test_match_states(self):
        self.assertFalse(
            np.allclose(self.state_time_course_1, self.state_time_course_2),
            msg="State time courses are the same.",
        )
        self.assertTrue(
            np.allclose(
                *array_ops.match_states(
                    self.state_time_course_1, self.state_time_course_2
                )
            ),
            msg="States post-matching were not the same.",
        )


class TestDiceCoefficient1D(TestCase):
    def setUp(self) -> None:
        self.sequence_1 = np.random.randint(5, size=1000)
        self.sequence_2 = self.sequence_1.copy()
        self.sequence_3 = np.random.permutation(self.sequence_1)

    def test_dice_coefficient_1d(self):
        dice_1 = array_ops.dice_coefficient_1d(self.sequence_1, self.sequence_2)
        dice_2 = array_ops.dice_coefficient_1d(self.sequence_1, self.sequence_3)

        self.assertTrue(dice_1 == 1, msg="dice(a, a) == 1.")
        self.assertTrue(
            0.1 < dice_2 < 0.3,
            msg="A randomly shuffled array should "
            "have a dice coefficient of around 1/n states.",
        )


class TestDiceCoefficient(TestCase):
    def setUp(self) -> None:
        self.sequence_1 = array_ops.get_one_hot(np.random.randint(5, size=1000))
        self.sequence_2 = self.sequence_1.copy()
        self.sequence_3 = np.random.permutation(self.sequence_1)

    def test_dice_coefficient_1d(self):
        dice_1 = array_ops.dice_coefficient(self.sequence_1, self.sequence_2)
        dice_2 = array_ops.dice_coefficient(self.sequence_1, self.sequence_3)

        self.assertTrue(dice_1 == 1, msg="dice(a, a) == 1.")
        self.assertTrue(
            0.1 < dice_2 < 0.3,
            msg="A randomly shuffled array should "
            "have a dice coefficient of around 1/n states.",
        )


class TestAlignArrays(TestCase):
    def setUp(self) -> None:
        self.sequence = array_ops.get_one_hot(np.random.randint(5, size=1000))
        self.left = self.sequence[:900]
        self.right = self.sequence[100:]
        self.center = self.sequence[50:950]

    def test_align_arrays(self):
        left_aligned = array_ops.align_arrays(self.sequence, self.left)
        right_aligned = array_ops.align_arrays(
            self.sequence, self.right, alignment="right"
        )
        center_aligned = array_ops.align_arrays(
            self.sequence, self.center, alignment="center"
        )

        self.assertTrue(np.all(np.equal(*left_aligned, self.left)))
        self.assertTrue(np.all(np.equal(*right_aligned, self.right)))
        self.assertTrue(np.all(np.equal(*center_aligned, self.center)))
