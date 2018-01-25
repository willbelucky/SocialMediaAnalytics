# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""
import pandas as pd

from sklearn.linear_model import Ridge


def get_ridge_regression(x_train, y_train, x_test, alpha):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :param alpha:
    :return:
    """
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)

    y_prediction = model.predict(X=x_test)

    return pd.Series(y_prediction)


# An usage example
if __name__ == '__main__':
    from data.data_reader import get_training_data
    from data.data_combinator import get_combinations

    alpha = 1.0

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    x_train = get_combinations(x_train)
    x_val = get_combinations(x_val)
    y_val = y_val.reset_index(drop=True)

    y_prediction = get_ridge_regression(x_train, y_train, x_val, alpha)
    result = pd.concat([y_val, y_prediction], axis=1)
    print(result.head())
