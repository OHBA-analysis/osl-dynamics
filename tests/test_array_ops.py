from unittest import TestCase

import numpy as np
import vrad.inference.metrics
from vrad import array_ops


class TestGetOneHot(TestCase):
    def setUp(self) -> None:
        self.state_time_course_2d = np.random.rand(100, 5)
        self.state_time_course = self.state_time_course_2d.argmax(axis=1)

    def test_get_one_hot(self):
        one_hot = array_ops.get_one_hot(self.state_time_course)

        self.assertTrue(
            np.all(one_hot.argmax(axis=1) == self.state_time_course),
            msg="One hot did not reproduce the categorical sequence when inverted.",
        )

    def test_get_one_hot_2D(self):
        one_hot_1d = array_ops.get_one_hot(self.state_time_course)
        one_hot_2d = array_ops.get_one_hot(self.state_time_course_2d)
        one_hot_2d_states = array_ops.get_one_hot(self.state_time_course_2d, n_states=7)

        self.assertTrue(np.all(one_hot_1d == one_hot_2d))
        self.assertTrue(np.all(one_hot_1d == one_hot_2d_states[:, :5]))


class TestDiceCoefficient1D(TestCase):
    def setUp(self) -> None:
        self.sequence_1 = np.random.randint(5, size=1000)
        self.sequence_2 = self.sequence_1.copy()
        self.sequence_3 = np.random.permutation(self.sequence_1)

    def test_dice_coefficient_1d(self):
        dice_1 = vrad.inference.metrics.dice_coefficient_1d(
            self.sequence_1, self.sequence_2
        )
        dice_2 = vrad.inference.metrics.dice_coefficient_1d(
            self.sequence_1, self.sequence_3
        )

        self.assertTrue(dice_1 == 1, msg="dice(a, a) == 1.")
        self.assertTrue(
            0.1 < dice_2 < 0.3,
            msg="A randomly shuffled array should "
            "have a dice coefficient of around 1/n states.",
        )

        one_hot = array_ops.get_one_hot(self.sequence_1)

        with self.assertRaises(ValueError):
            vrad.inference.metrics.dice_coefficient_1d(self.sequence_1, one_hot)

        with self.assertRaises(TypeError):
            vrad.inference.metrics.dice_coefficient_1d(
                self.sequence_1, self.sequence_2.astype(np.float)
            ),


class TestDiceCoefficient(TestCase):
    def setUp(self) -> None:
        self.base = np.random.randint(5, size=1000)
        self.sequence_1 = array_ops.get_one_hot(self.base)
        self.sequence_2 = self.sequence_1.copy()
        self.sequence_3 = np.random.permutation(self.sequence_1)

    def test_dice_coefficient_2d(self):
        dice_1 = vrad.inference.metrics.dice_coefficient(
            self.sequence_1, self.sequence_2
        )
        dice_2 = vrad.inference.metrics.dice_coefficient(
            self.sequence_1, self.sequence_3
        )

        self.assertTrue(dice_1 == 1, msg="dice(a, a) == 1.")
        self.assertTrue(
            0.1 < dice_2 < 0.3,
            msg="A randomly shuffled array should "
            "have a dice coefficient of around 1/n states.",
        )

        with self.assertRaises(ValueError):
            vrad.inference.metrics.dice_coefficient(
                np.random.rand(3, 4, 5), np.random.rand(5, 6, 7)
            )

        self.assertTrue(
            vrad.inference.metrics.dice_coefficient(self.base, self.base.copy()) == 1
        )

        self.assertTrue(
            vrad.inference.metrics.dice_coefficient(self.base, self.sequence_1) == 1
        )
        self.assertTrue(
            vrad.inference.metrics.dice_coefficient(self.sequence_1, self.base) == 1
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

        with self.assertRaises(ValueError):
            array_ops.align_arrays(self.sequence, self.left, alignment="problem")


class TestFromCholesky(TestCase):
    def setUp(self) -> None:
        r = np.random.rand(10, 10)
        self.psd = r @ r.T
        self.chol = np.linalg.cholesky(self.psd)

    def test_from_cholesky(self):
        reconstructed = array_ops.from_cholesky(self.chol)
        self.assertTrue(np.allclose(reconstructed, self.psd))

        reconstructed = array_ops.from_cholesky(self.chol[None])
        self.assertTrue(np.allclose(reconstructed, self.psd[None]))


class TestCalculateTransProbMatrix(TestCase):
    def setUp(self) -> None:
        rand = np.random.rand(5, 5)
        rand_norm = rand / rand.sum(axis=1)[:, None]
        trans_prob = rand_norm.cumsum(axis=1)

        state = [0]
        for i in range(1, 5000):
            r = np.random.rand()
            state.append(np.argmax(trans_prob[state[-1]] > r))

        self.state = np.array(state)
        z = np.zeros((5, 5))
        for i, j in zip(state, state[1:]):
            z[i, j] += 1

        self.z = z / z.sum(axis=1)[:, None]

    def test_calculate_trans_prob_matrix(self):
        inf_tb = array_ops.calculate_trans_prob_matrix(self.state)
        self.assertTrue(np.allclose(inf_tb, self.z))

        inf_tb_from_2d = array_ops.calculate_trans_prob_matrix(
            array_ops.get_one_hot(self.state)
        )

        self.assertTrue(np.allclose(inf_tb_from_2d, self.z))

        with self.assertRaises(ValueError):
            array_ops.calculate_trans_prob_matrix(np.random.rand(3, 6, 8))

        self.assertTrue(
            np.allclose(
                array_ops.calculate_trans_prob_matrix(self.state, n_states=10)[:5, :5],
                self.z,
            )
        )

        z2 = self.z.copy()
        np.fill_diagonal(z2, 0)
        self.assertTrue(
            np.allclose(
                array_ops.calculate_trans_prob_matrix(self.state, zero_diagonal=True),
                z2,
            )
        )


class TestTraceNormalize(TestCase):
    def setUp(self) -> None:
        self.rand2d = np.random.rand(10, 10)
        self.normed2d = self.rand2d / self.rand2d.trace()

        self.rand3d = np.random.rand(10, 10, 4)
        self.normed3d = self.rand3d / self.rand3d.trace(axis1=1, axis2=2)[:, None, None]

    def test_trace_normalize(self):
        self.assertTrue(
            np.allclose(self.normed2d, array_ops.trace_normalize(self.rand2d))
        )
        self.assertTrue(
            np.allclose(self.normed3d, array_ops.trace_normalize(self.rand3d))
        )

        with self.assertRaises(ValueError):
            array_ops.trace_normalize(np.random.rand(3, 4, 5, 6))


class Test(TestCase):
    def test_mean_diagonal(self):
        rand = np.random.rand(5, 5)
        result = array_ops.mean_diagonal(rand)
        self.assertTrue(
            np.allclose(result[~np.eye(5, dtype=bool)], rand[~np.eye(5, dtype=bool)])
        )
        self.assertTrue(
            np.allclose(
                result[np.eye(5, dtype=bool)], rand[~np.eye(5, dtype=bool)].mean()
            )
        )


class TestConfusionMatrix(TestCase):
    def test_confusion_matrix(self):
        a = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        b = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
        exp_result = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        self.assertTrue(
            np.all(exp_result == vrad.inference.metrics.confusion_matrix(a, b))
        )
