# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression, Lasso
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

COLUMN_NAME = 'column_name'
COEFFICIENT_VALUE = 'coefficient_value'


def get_logistic_regression(x_train, y_train, x_test):
    """

    :param x_train: (DataFrame) The variables of train set.
    :param y_train: (Series) The correct answers of train set.
    :param x_test: (DataFrame) The variables of test set.

    :return y_prediction: (Series) The predictions of test set.
    """
    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_prediction = model.predict(X=x_test)
    y_prediction = pd.Series(y_prediction)

    return y_prediction


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

    :return y_prediction: (Series) The predictions of test set.
    """
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)

    if summary:
        model_coef = pd.DataFrame(data=list(zip(x_train.columns, np.abs(model.coef_))),
                                  columns=[COLUMN_NAME, COEFFICIENT_VALUE])
        model_coef = model_coef.sort_values(by=COEFFICIENT_VALUE, ascending=False)
        print(model_coef)

    y_prediction = model.predict(X=x_test)
    y_prediction = pd.Series(y_prediction)

    return y_prediction


def get_lasso_regression(x_train, y_train, x_test, alpha, summary=False):
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

    :return y_prediction: (Series) The predictions of test set.
    """
    model = Lasso(alpha=alpha)
    model.fit(X=x_train, y=y_train)

    if summary:
        model_coef = pd.DataFrame(data=list(zip(x_train.columns, np.abs(model.coef_))),
                                  columns=[COLUMN_NAME, COEFFICIENT_VALUE])
        model_coef = model_coef.sort_values(by=COEFFICIENT_VALUE, ascending=False)
        print(model_coef)

    y_prediction = model.predict(x_test)
    y_prediction = pd.Series(y_prediction)

    return y_prediction


def get_linear_discriminant_analysis(x_train, y_train, x_test):
    """
    :param x_train:
    :param y_train:
    :param x_test:
    :return:
    """

    model = lda()
    model.fit(x_train, y_train)

    y_prediction = model.predict(X=x_test)
    y_prediction = pd.Series(y_prediction)
    # LDA_y_score = LDA.fit(x_train, y_train).decision_function(x_test)
    # fpr_LDA, tpr_LDA, threashold_LDA = roc_curve(y_test, LDA_y_score)
    return y_prediction

def get_quadratic_discriminant_analysis(x_train, y_train, x_test):
    """
    :param x_train:
    :param y_train:
    :param x_test:
    :return:
    """

    model = qda()
    model.fit(x_train, y_train)
    y_prediction = model.predict(X=x_test)
    y_prediction = pd.Series(y_prediction)
    # QDA_y_score = QDA.fit(x_train, y_train).decision_function(x_test)
    # fpr_QDA, tpr_QDA, threashold_QDA = roc_curve(y_test, QDA_y_score)
    return y_prediction

def get_naive_bayes(x_train, y_train, x_test):
    """
    :param x_train:
    :param y_train:
    :param x_test:
    :return:
    """

    model = GaussianNB()
    model.fit(x_train, y_train)
    y_prediction = model.predict(X=x_test)
    y_prediction = pd.Series(y_prediction)
    # GNB_y_score = GNB.fit(x_train, y_train).predict_proba(x_test)
    # fpr_GNB, tpr_GNB, threashold_GNB = roc_curve(y_test, -GNB_y_score[:0])
    return y_prediction


# An usage example
if __name__ == '__main__':
    from data.data_reader import get_training_data
    from data.data_combinator import get_full_combinations

    alpha = 0.001

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    x_train = get_full_combinations(x_train)
    x_val = get_full_combinations(x_val)
    y_val = y_val.reset_index(drop=True)
    #
    # prnt('Logistic Regression')
    # y_prediction = get_logistic_regression(x_train, y_train, x_val)
    # result = pd.concat([y_val, y_prediction], axis=1)
    # print(result.head())
    # print('-' * 70)
    #
    # print('Ridge Regression')
    # y_prediction = get_ridge_regression(x_train, y_train, x_val, alpha)
    # result = pd.concat([y_val, y_predi6y7`QDction], axis=1)
    # print(result.head())
    # print('-' * 70)
    #
    # print('Lasso Regression')
    # y_prediction = get_lasso_regression(x_train, y_train, x_val, alpha, True)
    # result = pd.concat([y_val, y_prediction], axis=1)
    # print(result.head())
    # print('-' * 70)

    print('LDA')
    y_prediction = get_linear_discriminant_analysis(x_train, y_train, x_val)
    result = pd.concat([y_val, y_prediction], axis=1)
    print(result.head())
    print('-' * 70)

    print('QDA')
    y_prediction = get_quadratic_discriminant_analysis(x_train, y_train, x_val)
    result = pd.concat([y_val, y_prediction], axis=1)
    print(result.head())
    print('-' * 70)

    print('GNB')
    y_prediction = get_naive_bayes(x_train, y_train, x_val)
    result = pd.concat([y_val, y_prediction], axis=1)
    print(result.head())
    print('-' * 70)
