# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_combinator import get_full_combinations, get_sub_combinations, get_div_combinations, \
    get_log_div_combinations
from data.data_reader import get_training_data
from evaluation.evaluator import evaluate_predictions
from stats.regression_calculator import get_ridge_regression, get_logistic_regression, get_lasso_regression, \
    get_linear_discriminant_analysis, get_quadratic_discriminant_analysis, get_naive_bayes

# A key for index value.
ALPHA = 'alpha'

# Keys for logistic regression.
LOGISTIC_AUC = 'logistic_auc'

# Keys for ridge regression.
RIDGE_AUC = 'ridge_auc'

# Keys for lasso regression.
LASSO_AUC = 'lasso_auc'

# Keys for lda.
LDA_AUC = 'lda_auc'

# Keys for qda.
QDA_AUC = 'qda_auc'

# Keys for gnb.
GNB_AUC = 'gnb_auc'

REGRESSION_COMPARISON_AUCS = [LOGISTIC_AUC, RIDGE_AUC, LASSO_AUC, LDA_AUC, QDA_AUC, GNB_AUC]


def draw_regression_comparison_graph(from_alpha, to_alpha, step):
    """
    This method shows you AUC(Area Under the Curve) of all regression methods.

    :param from_alpha: (float) from_alpha must be bigger than 0.
    :param to_alpha: (float) to_alpha must be bigger than from_alpha
    :param step: (float) The size of one step.
    """
    assert 0 < from_alpha
    assert from_alpha < to_alpha

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    x_train = get_full_combinations(x_train)
    x_val = get_full_combinations(x_val)

    # Dictionary for saving results.
    evaluation_results_dict = {ALPHA: []}
    for auc in REGRESSION_COMPARISON_AUCS:
        evaluation_results_dict[auc] = []

    # Logistic regression
    logistic_y_prediction = get_logistic_regression(x_train, y_train, x_val)
    _, _, logistic_auc = evaluate_predictions(y_val, logistic_y_prediction)

    # LDA
    lda_y_prediction = get_linear_discriminant_analysis(x_train, y_train, x_val)
    _, _, lda_auc = evaluate_predictions(y_val, lda_y_prediction)

    # QDA
    qda_y_prediction = get_quadratic_discriminant_analysis(x_train, y_train, x_val)
    _, _, qda_auc = evaluate_predictions(y_val, qda_y_prediction)

    # GNB
    gnb_y_prediction = get_naive_bayes(x_train, y_train, x_val)
    _, _, gnb_auc = evaluate_predictions(y_val, gnb_y_prediction)

    for alpha in np.arange(from_alpha, to_alpha, step):
        # Ridge regression
        ridge_y_prediction = get_ridge_regression(x_train, y_train, x_val, alpha)
        _, _, ridge_auc = evaluate_predictions(y_val, ridge_y_prediction)

        # Lasso regression
        lasso_y_prediction = get_lasso_regression(x_train, y_train, x_val, alpha)
        _, _, lasso_auc = evaluate_predictions(y_val, lasso_y_prediction)

        # Save index values.
        evaluation_results_dict[ALPHA].append(alpha)

        # Save results of logistic regression.
        evaluation_results_dict[LOGISTIC_AUC].append(logistic_auc)

        # Save results of ridge regression.
        evaluation_results_dict[RIDGE_AUC].append(ridge_auc)

        # Save results of lasso regression.
        evaluation_results_dict[LASSO_AUC].append(lasso_auc)

        # Save results of lda.
        evaluation_results_dict[LDA_AUC].append(lda_auc)

        # Save results of qda.
        evaluation_results_dict[QDA_AUC].append(qda_auc)

        # Save results of gnb.
        evaluation_results_dict[GNB_AUC].append(gnb_auc)

    evaluation_results_df = pd.DataFrame(data=evaluation_results_dict)

    # Print peek points
    for auc in REGRESSION_COMPARISON_AUCS:
        highest_auc_row = evaluation_results_df.loc[evaluation_results_df[auc].idxmax(), [auc, ALPHA]]
        print('The highest {}={} when alpha={}'.format(auc, highest_auc_row[auc], highest_auc_row[ALPHA]))

    evaluation_results_df = evaluation_results_df.set_index([ALPHA])
    evaluation_results_df.plot(title='Logistic vs. Ridge vs. Lasso vs. GNB vs. LDA vs. QDA', grid=True, ylim=(0.0, 1))
    plt.show()


# Keys for full combined ridge regression.
FULL_COMBINED_AUC = 'full_combined_auc'

# Keys for sub combined ridge regression.
SUB_COMBINED_AUC = 'sub_combined_auc'

# Keys for div combined ridge regression.
DIV_COMBINED_AUC = 'div_combined_auc'

# Keys for log div combined ridge regression.
LOG_DIV_COMBINED_AUC = 'log_div_combined_auc'

# Keys for log div combined ridge regression.
ROOT_DIV_COMBINED_AUC = 'root_div_combined_auc'

COMBINATION_COMPARISON_AUCS = [FULL_COMBINED_AUC, SUB_COMBINED_AUC, DIV_COMBINED_AUC, LOG_DIV_COMBINED_AUC,
                               ROOT_DIV_COMBINED_AUC]


def draw_combination_comparison_graph(regression_function, function_name, from_alpha, to_alpha, step):
    """

    :param regression_function: The regression function you want to see the graph.
    :param function_name: The name of the function.
    :param from_alpha: (float) from_alpha must be bigger than 0.
    :param to_alpha: (float) to_alpha must be bigger than from_alpha
    :param step: (float) The size of one step.
    """
    assert 0 < from_alpha
    assert from_alpha < to_alpha

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    full_combined_x_train = get_full_combinations(x_train)
    full_combined_x_val = get_full_combinations(x_val)
    sub_combined_x_train = get_sub_combinations(x_train)
    sub_combined_x_val = get_sub_combinations(x_val)
    div_combined_x_train = get_div_combinations(x_train)
    div_combined_x_val = get_div_combinations(x_val)
    log_div_combined_x_train = get_log_div_combinations(x_train)
    log_div_combined_x_val = get_log_div_combinations(x_val)
    root_div_combined_x_train = get_log_div_combinations(x_train)
    root_div_combined_x_val = get_log_div_combinations(x_val)

    # Dictionary for saving results.
    evaluation_results_dict = {ALPHA: []}
    for auc in COMBINATION_COMPARISON_AUCS:
        evaluation_results_dict[auc] = []

    for alpha in np.arange(from_alpha, to_alpha, step):
        # Full combined ridge regression
        full_combined_ridge_y_prediction = \
            regression_function(full_combined_x_train, y_train, full_combined_x_val, alpha)
        _, _, full_combined_auc = evaluate_predictions(y_val, full_combined_ridge_y_prediction)

        # Sub combined ridge regression
        sub_combined_ridge_y_prediction = \
            regression_function(sub_combined_x_train, y_train, sub_combined_x_val, alpha)
        _, _, sub_combined_auc = evaluate_predictions(y_val, sub_combined_ridge_y_prediction)

        # Div combined ridge regression
        div_combined_ridge_y_prediction = \
            regression_function(div_combined_x_train, y_train, div_combined_x_val, alpha)
        _, _, div_combined_auc = evaluate_predictions(y_val, div_combined_ridge_y_prediction)

        # Log div combined ridge regression
        log_div_combined_ridge_y_prediction = \
            regression_function(log_div_combined_x_train, y_train, log_div_combined_x_val, alpha)
        _, _, log_div_combined_auc = evaluate_predictions(y_val, log_div_combined_ridge_y_prediction)

        # Log div combined ridge regression
        root_div_combined_ridge_y_prediction = \
            regression_function(root_div_combined_x_train, y_train, root_div_combined_x_val, alpha)
        _, _, root_div_combined_auc = evaluate_predictions(y_val, root_div_combined_ridge_y_prediction)

        # Save index values.
        evaluation_results_dict[ALPHA].append(alpha)

        # Save results of full combined ridge regression.
        evaluation_results_dict[FULL_COMBINED_AUC].append(full_combined_auc)

        # Save results of sub combined ridge regression.
        evaluation_results_dict[SUB_COMBINED_AUC].append(sub_combined_auc)

        # Save results of div combined ridge regression.
        evaluation_results_dict[DIV_COMBINED_AUC].append(div_combined_auc)

        # Save results of div combined ridge regression.
        evaluation_results_dict[LOG_DIV_COMBINED_AUC].append(log_div_combined_auc)

        # Save results of div combined ridge regression.
        evaluation_results_dict[ROOT_DIV_COMBINED_AUC].append(root_div_combined_auc)

    evaluation_results_df = pd.DataFrame(data=evaluation_results_dict)

    evaluation_results_df = evaluation_results_df.set_index([ALPHA])
    evaluation_results_df.plot(
        title='{}: Full vs. Sub vs. Div vs. LogDiv vs. RootDiv'.format(function_name),
        grid=True,
        ylim=(0.0, 1)
    )
    plt.show()


# An usage example
if __name__ == '__main__':
    draw_regression_comparison_graph(0.001, 0.200, 0.001)

    # Set the function you want to test.
    regression_function = get_lasso_regression
    # Set the name of the function.
    regression_function_name = 'Lasso regression'
    draw_combination_comparison_graph(regression_function, regression_function_name, 0.001, 0.200, 0.001)
