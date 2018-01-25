# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data.data_reader import get_training_data
from data.data_combinator import get_combinations
from stats.regression_calculator import get_ridge_regression
from evaluation.evaluator import evaluate_predictions


def draw_graph(from_alpha, to_alpha, step):
    assert from_alpha < to_alpha

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    x_train = get_combinations(x_train)
    x_val = get_combinations(x_val)

    evaluation_results_dict = {'alpha': [], 'accuracy': [], 'f1_score': [], 'MSE': []}
    for alpha in np.arange(from_alpha, to_alpha, step):
        y_prediction = get_ridge_regression(x_train, y_train, x_val, alpha)
        accuracy, f1_score, MSE = evaluate_predictions(y_val, y_prediction)
        evaluation_results_dict['alpha'].append(alpha)
        evaluation_results_dict['accuracy'].append(accuracy)
        evaluation_results_dict['f1_score'].append(f1_score)
        evaluation_results_dict['MSE'].append(MSE)

    evaluation_results_df = pd.DataFrame(data=evaluation_results_dict)
    evaluation_results_df = evaluation_results_df.set_index(['alpha'])
    evaluation_results_df.plot(title='Ridge Regression', grid=True, secondary_y=['MSE'], ylim=(0.5, 1))
    plt.show()


# An usage example
if __name__ == '__main__':
    draw_graph(0.01, 1.00, 0.01)
