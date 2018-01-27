# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix as cm


def plot_confusion_matrix(confusion_matrix, class_names=None,
                          normalize=False,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if class_names is None:
        class_names = ['False', 'True']

    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(confusion_matrix)

    plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def custom_round(number):
    return 1 if number >= 0.5 else 0


# noinspection PyPep8Naming
def evaluate_predictions(y_actual, y_prediction, confusion_matrix_plotting=False):
    """

    :param y_actual: (Series) The actual y values.
    :param y_prediction: (Series) The predicted y values.
    :param confusion_matrix_plotting: (bool) If confusion_matrix_plotting is True, plot the confusion matrix.

    :return accuracy: (float) The portion of correct predictions.
    :return f1_score: (float) The harmonic mean of precision and recall
    :return MSE: (float) The mean squared error.
    """
    y_actual = y_actual.reset_index(drop=True)
    y_prediction = y_prediction.reset_index(drop=True)
    assert len(y_actual) == len(y_prediction)

    y_prediction_round = y_prediction.apply(custom_round)
    confusion_matrix = cm(y_actual, y_prediction_round)
    TN, FP, FN, TP = confusion_matrix.ravel()
    Precision = TP / (TP + FP + 1e-20)  # The portion of actual 1 of prediction 1.
    Recall = TP / (TP + FN + 1e-20)  # The portion of prediction 1 of actual 1.
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # The portion of correct predictions.
    f1_score = 2 / (1 / Precision + 1 / Recall + 1e-20)  # The harmonic mean of precision and recall.

    if confusion_matrix_plotting:
        plot_confusion_matrix(confusion_matrix)

    return accuracy, f1_score


# An usage example.
if __name__ == '__main__':
    from data.data_reader import get_training_data
    from data.data_combinator import get_full_combinations
    from stats.regression_calculator import get_lasso_regression

    alpha = 0.001

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    x_train = get_full_combinations(x_train)
    x_val = get_full_combinations(x_val)

    print('Ridge regression')
    y_prediction = get_lasso_regression(x_train, y_train, x_val, alpha)
    accuracy, f1_score = evaluate_predictions(y_val, y_prediction, confusion_matrix_plotting=True)
    print('accuracy:{}'.format(accuracy))
    print('f1_score:{}'.format(f1_score))
    print('-' * 70)
