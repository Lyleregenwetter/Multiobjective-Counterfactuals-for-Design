import unittest

from pymoo.core.mixed import MixedVariableDuplicateElimination

from decode_mcd_private.efficient_mixed_duplicate_elimination import EfficientMixedVariableDuplicateElimination


class ValueHolder:
    def __init__(self, inner_dict: dict):
        self.X = inner_dict


class MixedVariableDuplicateEliminationTest(unittest.TestCase):
    def setUp(self):
        self.original = MixedVariableDuplicateElimination()
        self.improved = EfficientMixedVariableDuplicateElimination()

    def test_is_equal(self):
        first_object = ValueHolder({1: 1, 2: 2})
        second_object = ValueHolder({2: 2, 1: 1})
        self.assertTrue(self.original.is_equal(first_object, second_object))
        self.assertTrue(self.improved.is_equal(first_object, second_object))

        self.assertEqual(self.original.is_equal(first_object, second_object),
                         self.improved.is_equal(first_object, second_object))

    def test_is_not_equal(self):
        first = ValueHolder({1: 1})
        second = ValueHolder({1: 2})

        self.assertFalse(self.original.is_equal(first, second))
        self.assertFalse(self.improved.is_equal(first, second))

        first = ValueHolder({1: 2})
        second = ValueHolder({2: 2})
        self.assertFalse(self.original.is_equal(first, second))
        self.assertFalse(self.improved.is_equal(first, second))

    def test_old_does_subset_check(self):
        """The original inefficient pymoo method does a subset check,
        but it doesn't seem to matter and might well be unintentional"""
        first = ValueHolder({1: 1})
        second = ValueHolder({1: 1, 2: 2})

        self.assertTrue(self.original.is_equal(first, second))
        self.assertFalse(self.improved.is_equal(first, second))
