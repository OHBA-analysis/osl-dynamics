from unittest import TestCase

from dynemo.utils import misc


class TestListify(TestCase):
    def test_listify(self):
        objects = ["string", 10, True, [], 7.3, None]
        for obj in objects:
            listified = misc.listify(obj)
            self.assertTrue(isinstance(listified, list))
            if isinstance(obj, list):
                self.assertTrue(obj == listified)
            elif isinstance(obj, type(None)):
                self.assertTrue(len(listified) == 0)
            else:
                self.assertTrue(len(listified) == 1)
                self.assertTrue(isinstance(listified[0], type(obj)))
                self.assertTrue(obj == listified[0])


class TestDictOverride(TestCase):
    def test_override_dict_defaults(self):
        default_dict = {"a": 1, "b": 2}
        overrides = [
            {},
            None,
            {"c": 3},
        ]
        expected_results = [default_dict, default_dict, {"a": 1, "b": 2, "c": 3}]

        for override, expected_result in zip(overrides, expected_results):
            self.assertTrue(
                misc.override_dict_defaults(default_dict, override) == expected_result
            )
