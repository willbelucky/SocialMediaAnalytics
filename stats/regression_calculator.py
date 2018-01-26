# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, LogisticRegression


def get_logistic_regression(x_train, y_train, x_test):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :return:
    """
    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_prediction = model.predict(X=x_test)

    return pd.Series(y_prediction)


def get_ridge_regression(x_train, y_train, x_test, alpha, summary=False):
    """

    :param x_train: (DataFrame) The variables of train set.
    :param y_train: (Series) The correct answers of train set.
    :param x_test: (DataFrame) The variables of test set.
    :param alpha: Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``C^-1`` in other linear models such as
        LogisticRegression or LinearSVC. If an array is passed, penalties are
        assumed to be specific to the targets. Hence they must correspond in
        number.
    :param summary: (bool) If summary is True, print the coefficient values by descent order.
    :return:
    """
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)

    if summary:
        model_coef = pd.DataFrame(data=list(zip(x_train.columns, np.abs(model.coef_))),
                                  columns=['column_name', 'coefficient_value'])
        model_coef = model_coef.sort_values(by='coefficient_value', ascending=False)
        print(model_coef)

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

    print('Logistic Regression')
    y_prediction = get_logistic_regression(x_train, y_train, x_val)
    result = pd.concat([y_val, y_prediction], axis=1)
    print(result.head())
    print('-' * 70)

    print('Ridge Regression')
    y_prediction = get_ridge_regression(x_train, y_train, x_val, alpha)
    result = pd.concat([y_val, y_prediction], axis=1)
    print(result.head())
    print('-' * 70)
