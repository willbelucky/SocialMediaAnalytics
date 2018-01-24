# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 23.
"""
from unittest import TestCase
from data.data_reader import *


class TestGetTrainingData(TestCase):
    def test_get_training_data_with_no_validation(self):
        x_train, y_train, x_val, y_val = get_training_data()

        self.assertIsNotNone(x_train)
        self.assertEqual(5500, len(x_train))
        self.assertEqual(22, len(x_train.columns))

        self.assertIsNotNone(y_train)
        self.assertEqual(5500, len(y_train))

        self.assertIsNone(x_val)

        self.assertIsNone(y_val)

    def test_get_training_data_with_validation(self):
        x_train, y_train, x_val, y_val = get_training_data(validation=True, validation_size=0.4)

        self.assertIsNotNone(x_train)
        self.assertEqual(3300, len(x_train))
        self.assertEqual(22, len(x_train.columns))

        self.assertIsNotNone(y_train)
        self.assertEqual(3300, len(y_train))

        self.assertIsNotNone(x_val)
        self.assertEqual(2200, len(x_val))
        self.assertEqual(22, len(x_val.columns))

        self.assertIsNotNone(y_val)
        self.assertEqual(2200, len(y_val))

    def test_get_training_data_with_too_small_validation_size(self):
        with self.assertRaises(ValueError):
            get_training_data(validation=True, validation_size=-0.1)

    def test_get_training_data_with_too_big_validation_size(self):
        with self.assertRaises(ValueError):
            get_training_data(validation=True, validation_size=1.1)


class TestGetTestingData(TestCase):
    def test_get_training_data(self):
        x_test = get_test_data()

        self.assertIsNotNone(x_test)
        self.assertEqual(5952, len(x_test))
        self.assertEqual(22, len(x_test.columns))
