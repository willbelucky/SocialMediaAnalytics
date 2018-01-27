# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_combinator import get_full_combinations, get_sub_combinations, get_div_combinations
from data.data_reader import get_training_data
from evaluation.evaluator import evaluate_predictions
from stats.regression_calculator import get_ridge_regression, get_logistic_regression, get_lasso_regression, \
    get_linear_discriminant_analysis, get_quadratic_discriminant_analysis, get_naive_bayes

# A key for index value.
ALPHA = 'alpha'

# Keys for logistic regression.
LOGISTIC_F1_SCORE = 'logistic_f1_score'

# Keys for ridge regression.
RIDGE_F1_SCORE = 'ridge_f1_score'

# Keys for lasso regression.
LASSO_F1_SCORE = 'lasso_f1_score'

# Keys for lda.
LDA_F1_SCORE = 'lda_f1_score'

# Keys for qda.
QDA_F1_SCORE = 'qda_f1_score'

# Keys for gnb.
GNB_F1_SCORE = 'gnb_f1_score'


def draw_regression_comparison_graph(from_alpha, to_alpha, step):
    assert from_alpha < to_alpha

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    x_train = get_full_combinations(x_train)
    x_val = get_full_combinations(x_val)

    # Dictionary for saving results.
    evaluation_results_dict = {
        ALPHA: [],
        LOGISTIC_F1_SCORE: [],
        RIDGE_F1_SCORE: [],
        LASSO_F1_SCORE: [],
        LDA_F1_SCORE: [],
        QDA_F1_SCORE: [],
        GNB_F1_SCORE: [],
    }

    # Logistic regression
    logistic_y_prediction = get_logistic_regression(x_train, y_train, x_val)
    _, logistic_f1_score = evaluate_predictions(y_val, logistic_y_prediction)

    # LDA
    lda_y_prediction = get_linear_discriminant_analysis(x_train, y_train, x_val)
    _, lda_f1_score = evaluate_predictions(y_val, lda_y_prediction)

    # QDA
    qda_y_prediction = get_quadratic_discriminant_analysis(x_train, y_train, x_val)
    _, qda_f1_score = evaluate_predictions(y_val, qda_y_prediction)

    # GNB
    gnb_y_prediction = get_naive_bayes(x_train, y_train, x_val)
    _, gnb_f1_score = evaluate_predictions(y_val, gnb_y_prediction)

    for alpha in np.arange(from_alpha, to_alpha, step):
        # Ridge regression
        ridge_y_prediction = get_ridge_regression(x_train, y_train, x_val, alpha)
        _, ridge_f1_score = evaluate_predictions(y_val, ridge_y_prediction)

        # Lasso regression
        lasso_y_prediction = get_lasso_regression(x_train, y_train, x_val, alpha)
        _, lasso_f1_score = evaluate_predictions(y_val, lasso_y_prediction)

        # Save index values.
        evaluation_results_dict[ALPHA].append(alpha)

        # Save results of logistic regression.
        evaluation_results_dict[LOGISTIC_F1_SCORE].append(logistic_f1_score)

        # Save results of ridge regression.
        evaluation_results_dict[RIDGE_F1_SCORE].append(ridge_f1_score)

        # Save results of lasso regression.
        evaluation_results_dict[LASSO_F1_SCORE].append(lasso_f1_score)

        # Save results of lda.
        evaluation_results_dict[LDA_F1_SCORE].append(lda_f1_score)

        # Save results of qda.
        evaluation_results_dict[QDA_F1_SCORE].append(qda_f1_score)

        # Save results of gnb.
        evaluation_results_dict[GNB_F1_SCORE].append(gnb_f1_score)

    evaluation_results_df = pd.DataFrame(data=evaluation_results_dict)

    # Print peek points
    highest_f1_score_row = \
        evaluation_results_df.loc[evaluation_results_df[RIDGE_F1_SCORE].argmax(), [RIDGE_F1_SCORE, ALPHA]]
    print('The highest f1_score={} when alpha={}'
          .format(highest_f1_score_row[RIDGE_F1_SCORE], highest_f1_score_row[ALPHA]))

    evaluation_results_df = evaluation_results_df.set_index([ALPHA])
    evaluation_results_df.plot(title='Logistic vs. Ridge vs. Lasso vs. GNB vs. LDA vs. QDA', grid=True, ylim=(0.0, 1))
    plt.show()


# Keys for full combined ridge regression.
FULL_COMBINED_RIDGE_F1_SCORE = 'full_combined_ridge_f1_score'

# Keys for sub combined ridge regression.
SUB_COMBINED_RIDGE_F1_SCORE = 'sub_combined_ridge_f1_score'

# Keys for div combined ridge regression.
DIV_COMBINED_RIDGE_F1_SCORE = 'div_combined_ridge_f1_score'


def draw_combination_comparison_graph(from_alpha, to_alpha, step):
    assert from_alpha < to_alpha

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    full_combined_x_train = get_full_combinations(x_train)
    full_combined_x_val = get_full_combinations(x_val)
    sub_combined_x_train = get_sub_combinations(x_train)
    sub_combined_x_val = get_sub_combinations(x_val)
    div_combined_x_train = get_div_combinations(x_train)
    div_combined_x_val = get_div_combinations(x_val)

    # Dictionary for saving results.
    evaluation_results_dict = {
        ALPHA: [],
        FULL_COMBINED_RIDGE_F1_SCORE: [],
        SUB_COMBINED_RIDGE_F1_SCORE: [],
        DIV_COMBINED_RIDGE_F1_SCORE: [],
    }

    for alpha in np.arange(from_alpha, to_alpha, step):
        # Full combined ridge regression
        full_combined_ridge_y_prediction = \
            get_ridge_regression(full_combined_x_train, y_train, full_combined_x_val, alpha)
        _, full_combined_ridge_f1_score = evaluate_predictions(y_val, full_combined_ridge_y_prediction)

        # Sub combined ridge regression
        sub_combined_ridge_y_prediction = \
            get_ridge_regression(sub_combined_x_train, y_train, sub_combined_x_val, alpha)
        _, sub_combined_ridge_f1_score = evaluate_predictions(y_val, sub_combined_ridge_y_prediction)

        # Div combined ridge regression
        div_combined_ridge_y_prediction = \
            get_ridge_regression(div_combined_x_train, y_train, div_combined_x_val, alpha)
        _, div_combined_ridge_f1_score = evaluate_predictions(y_val, div_combined_ridge_y_prediction)

        # Save index values.
        evaluation_results_dict[ALPHA].append(alpha)

        # Save results of full combined ridge regression.
        evaluation_results_dict[FULL_COMBINED_RIDGE_F1_SCORE].append(full_combined_ridge_f1_score)

        # Save results of sub combined ridge regression.
        evaluation_results_dict[SUB_COMBINED_RIDGE_F1_SCORE].append(sub_combined_ridge_f1_score)

        # Save results of div combined ridge regression.
        evaluation_results_dict[DIV_COMBINED_RIDGE_F1_SCORE].append(div_combined_ridge_f1_score)

    evaluation_results_df = pd.DataFrame(data=evaluation_results_dict)

    evaluation_results_df = evaluation_results_df.set_index([ALPHA])
    evaluation_results_df.plot(title='Full vs. Sub vs. Div', grid=True, ylim=(0.5, 1))
    plt.show()


# An usage example
if __name__ == '__main__':
    draw_regression_comparison_graph(0.001, 0.200, 0.001)
    draw_combination_comparison_graph(0.01, 1.00, 0.01)
