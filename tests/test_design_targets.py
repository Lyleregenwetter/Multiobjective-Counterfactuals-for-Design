import unittest

from design_targets import ContinuousTarget


class DesignTargetsTest(unittest.TestCase):

    def test_invalid_continuous_target(self):
        factory_to_exception = {
            lambda: ContinuousTarget("LABEL", 50, None): "Upper bound must be a real number",
            lambda: ContinuousTarget("LABEL", None, 500): "Lower bound must be a real number",
            lambda: ContinuousTarget(None, 50, 500): "Label must be of type string or an integer index",
            lambda: ContinuousTarget("LABEL", 500, 50): "Lower bound cannot be greater or equal to upper bound",
            lambda: ContinuousTarget("LABEL", 50, 50): "Lower bound cannot be greater or equal to upper bound",
            lambda: ContinuousTarget("", 50, 500): "Label cannot be an empty string",
        }
        for factory, exception_message in factory_to_exception.items():
            with self.subTest():
                self.assert_raises_with_message(factory, exception_message)

    def test_valid_continuous_target(self):
        ContinuousTarget("LABEL", 50, 500)
        ContinuousTarget(5, 50, 500)
        ContinuousTarget(5, 50.5, 500.15)

    def assert_raises_with_message(self, faulty_call: callable, expected_message: str):
        with self.assertRaises(ValueError) as context:
            faulty_call()
        self.assertEqual(expected_message, context.exception.args[0])
