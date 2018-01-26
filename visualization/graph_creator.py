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
from stats.regression_calculator import get_ridge_regression, get_logistic_regression, get_lasso_regression

# A key for index value.
ALPHA = 'alpha'

# Keys for logistic regression.
LOGISTIC_ACCURACY = 'logistic_accuracy'
LOGISTIC_F1_SCORE = 'logistic_f1_score'

# Keys for ridge regression.
RIDGE_ACCURACY = 'ridge_accuracy'
RIDGE_F1_SCORE = 'ridge_f1_score'

# Keys for lasso regression.
LASSO_ACCURACY = 'lasso_accuracy'
LASSO_F1_SCORE = 'lasso_f1_score'


# noinspection PyPep8Naming
def draw_graph(from_alpha, to_alpha, step):
    assert from_alpha < to_alpha

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    x_train = get_combinations(x_train)
    x_val = get_combinations(x_val)

    # Dictionary for saving results.
    evaluation_results_dict = {ALPHA: [],
                               LOGISTIC_ACCURACY: [], LOGISTIC_F1_SCORE: [],
                               RIDGE_ACCURACY: [], RIDGE_F1_SCORE: [],
                               LASSO_ACCURACY: [], LASSO_F1_SCORE: []}

    # Logistic regression
    logistic_y_prediction = get_logistic_regression(x_train, y_train, x_val)
    logistic_accuracy, logistic_f1_score = evaluate_predictions(y_val, logistic_y_prediction)

    for alpha in np.arange(from_alpha, to_alpha, step):
        # Ridge regression
        ridge_y_prediction = get_ridge_regression(x_train, y_train, x_val, alpha)
        # noinspection PyPep8Naming
        ridge_accuracy, ridge_f1_score = evaluate_predictions(y_val, ridge_y_prediction)

        # Lasso regression
        lasso_y_prediction = get_lasso_regression(x_train, y_train, x_val, alpha)
        # noinspection PyPep8Naming
        lasso_accuracy, lasso_f1_score = evaluate_predictions(y_val, lasso_y_prediction)

        # Save index values.
        evaluation_results_dict[ALPHA].append(alpha)

        # Save results of logistic regression.
        evaluation_results_dict[LOGISTIC_ACCURACY].append(logistic_accuracy)
        evaluation_results_dict[LOGISTIC_F1_SCORE].append(logistic_f1_score)

        # Save results of ridge regression.
        evaluation_results_dict[RIDGE_ACCURACY].append(ridge_accuracy)
        evaluation_results_dict[RIDGE_F1_SCORE].append(ridge_f1_score)

        # Save results of lasso regression.
        evaluation_results_dict[LASSO_ACCURACY].append(lasso_accuracy)
        evaluation_results_dict[LASSO_F1_SCORE].append(lasso_f1_score)

    evaluation_results_df = pd.DataFrame(data=evaluation_results_dict)

    # Print peek points
    highest_f1_score_row = \
        evaluation_results_df.loc[evaluation_results_df[RIDGE_F1_SCORE].argmax(), [RIDGE_F1_SCORE, ALPHA]]
    print('The highest f1_score={} when alpha={}'
          .format(highest_f1_score_row[RIDGE_F1_SCORE], highest_f1_score_row[ALPHA]))
    highest_accuracy_row = \
        evaluation_results_df.loc[evaluation_results_df[RIDGE_ACCURACY].argmax(), [RIDGE_ACCURACY, ALPHA]]
    print('The highest accuracy={} when alpha={}'
          .format(highest_accuracy_row[RIDGE_ACCURACY], highest_accuracy_row[ALPHA]))

    evaluation_results_df = evaluation_results_df.set_index([ALPHA])
    evaluation_results_df.plot(title='Logistic vs. Ridge vs. Lasso', grid=True, ylim=(0.0, 1))
    plt.show()


# An usage example
if __name__ == '__main__':
    draw_graph(0.001, 0.200, 0.001)
