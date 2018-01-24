# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""
from unittest import TestCase

from data.data_reader import get_training_data
from data.data_combinator import get_combinations


class TestGetCombinations(TestCase):
    def test_get_combinations(self):
        x_train, _, _, _ = get_training_data()
        x_train = get_combinations(x_train)

        self.assertIsNotNone(x_train)
        self.assertEqual(5500, len(x_train))
        self.assertEqual(22, len(x_train.columns))
