# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""


# noinspection PyPep8Naming
def evaluate_predictions(y_actual, y_prediction):
    """

    :param y_actual: (Series) The actual y values.
    :param y_prediction: (Series) The predicted y values.

    :return accuracy: (float) The portion of correct predictions.
    :return f1_score: (float) The harmonic mean of precision and recall
    :return MSE: (float) The mean squared error.
    """
    y_actual = y_actual.reset_index(drop=True)
    y_prediction = y_prediction.reset_index(drop=True)
    assert len(y_actual) == len(y_prediction)

    y_prediction_round = y_prediction.apply(round)
    TP = sum(y_actual * y_prediction_round)  # The number of actual 1 predicted 1.
    TN = sum((y_actual - 1) * (y_prediction_round - 1))  # The number of actual 0 predicted 0.
    FP = -1 * sum((y_actual - 1) * y_prediction_round)  # The number of actual 0 predicted 1.
    FN = -1 * sum(y_actual * (y_prediction_round - 1))  # The number of actual 1 predicted 0.
    Precision = TP / (TP + FP + 1e-20)  # The portion of actual 1 of prediction 1.
    Recall = TP / (TP + FN + 1e-20)  # The portion of prediction 1 of actual 1.
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # The portion of correct predictions.
    f1_score = 2 / (1 / Precision + 1 / Recall + 1e-20)  # The harmonic mean of precision and recall.

    return accuracy, f1_score


# An usage example.
if __name__ == '__main__':
    from data.data_reader import get_training_data
    from data.data_combinator import get_combinations
    from stats.regression_calculator import get_ridge_regression

    alpha = 1.0

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    x_train = get_combinations(x_train)
    x_val = get_combinations(x_val)
    y_prediction = get_ridge_regression(x_train, y_train, x_val, alpha)

    accuracy, f1_score = evaluate_predictions(y_val, y_prediction)
    print('accuracy:{}'.format(accuracy))
    print('f1_score:{}'.format(f1_score))
