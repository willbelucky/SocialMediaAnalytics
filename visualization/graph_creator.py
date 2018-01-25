# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_combinator import get_combinations
from data.data_reader import get_training_data
from evaluation.evaluator import evaluate_predictions
from stats.regression_calculator import get_ridge_regression, get_logistic_regression

# A key for index value.
ALPHA = 'alpha'

# Keys for logistic regression.
LOGISTIC_ACCURACY = 'logistic_accuracy'
LOGISTIC_F1_SCORE = 'logistic_f1_score'

# Keys for ridge regression.
RIDGE_ACCURACY = 'ridge_accuracy'
RIDGE_F1_SCORE = 'ridge_f1_score'


def draw_graph(from_alpha, to_alpha, step):
    assert from_alpha < to_alpha

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    x_train = get_combinations(x_train)
    x_val = get_combinations(x_val)

    # Dictionary for saving results.
    evaluation_results_dict = {ALPHA: [],
                               LOGISTIC_ACCURACY: [], LOGISTIC_F1_SCORE: [],
                               RIDGE_ACCURACY: [], RIDGE_F1_SCORE: []}

    # Logistic regression
    y_prediction = get_logistic_regression(x_train, y_train, x_val)
    logistic_accuracy, logistic_f1_score, logistic_MSE = evaluate_predictions(y_val, y_prediction)

    # Ridge regression
    for alpha in np.arange(from_alpha, to_alpha, step):
        y_prediction = get_ridge_regression(x_train, y_train, x_val, alpha)
        # noinspection PyPep8Naming
        ridge_accuracy, ridge_f1_score, ridge_MSE = evaluate_predictions(y_val, y_prediction)

        # Save index values.
        evaluation_results_dict[ALPHA].append(alpha)

        # Save results of logistic regression.
        evaluation_results_dict[LOGISTIC_ACCURACY].append(logistic_accuracy)
        evaluation_results_dict[LOGISTIC_F1_SCORE].append(logistic_f1_score)

        # Save results of ridge regression.
        evaluation_results_dict[RIDGE_ACCURACY].append(ridge_accuracy)
        evaluation_results_dict[RIDGE_F1_SCORE].append(ridge_f1_score)

    evaluation_results_df = pd.DataFrame(data=evaluation_results_dict)
    evaluation_results_df = evaluation_results_df.set_index([ALPHA])
    evaluation_results_df.plot(title='Logistic Regression vs. Ridge Regression', grid=True, ylim=(0.5, 1))
    plt.show()


# An usage example
if __name__ == '__main__':
    draw_graph(0.01, 1.00, 0.01)
