from taser import array_ops
import numpy as np
from unittest import TestCase


class TestCorrelation(TestCase):
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
