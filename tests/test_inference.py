from unittest import TestCase

import numpy as np
from vrad import array_ops
from vrad import inference


class TestCorrelateStates(TestCase):
    def setUp(self) -> None:
        self.state_time_course_1 = array_ops.get_one_hot(
            np.random.randint(10, size=1000)
        )
        self.state_time_course_2 = self.state_time_course_1.copy()

    def test_correlate_states(self):
        correlation = inference.states.correlate_states(
            self.state_time_course_1, self.state_time_course_2
        )

        self.assertTrue(np.allclose(np.diagonal(correlation), 1))


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
                *inference.states.match_states(
                    self.state_time_course_1, self.state_time_course_2
                )
            ),
            msg="States post-matching were not the same.",
        )


class TestStateActivation(TestCase):
    def setUp(self) -> None:
        r2 = np.random.randint(5, size=1000)
        to_delete = []
        for idx, (i, j) in enumerate(zip(r2, r2[1:])):
            if i == j:
                to_delete.append(idx + 1)

        self.r2 = np.delete(r2, to_delete)

        self.r = np.random.randint(5, 10, size=self.r2.size)

        activation = []
        for i, j in zip(self.r, self.r2):
            activation.extend([j] * i)
        activation = np.array(activation)

        self.one_hot = array_ops.get_one_hot(activation, n_states=7)

    def test_state_activation(self):
        ons, offs = inference.states.state_activation(self.one_hot)
        for idx, (on, off) in enumerate(zip(ons, offs)):
            self.assertTrue(np.all((off - on) == self.r[self.r2 == idx]))

    def test_state_lifetimes(self):
        self.assertTrue(
            np.concatenate(
                [
                    i == self.r[self.r2 == j]
                    for i, j in zip(
                        inference.states.state_lifetimes(self.one_hot), range(5)
                    )
                ]
            ).all()
        )


class TestReduceStateTimeCourse(TestCase):
    def setUp(self) -> None:
        self.a = array_ops.get_one_hot(np.random.randint(5, size=1000))
        self.b = np.zeros((self.a.shape[0], self.a.shape[1] + 1))
        self.b[:, :-1] = self.a

    def test_reduce_state_time_course(self):
        self.assertTrue(
            np.all(inference.states.reduce_state_time_course(self.b) == self.a)
        )
