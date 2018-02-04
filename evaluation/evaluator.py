# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""
import matplotlib.pyplot as plt
import pandas as pd
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import roc_curve, auc

from util.plot_supporter import get_cmaps


# noinspection PyPep8Naming
def plot_roc_curve(fpr, tpr, AUC, title=None, label=None, color='darkorange'):
    """

    :param fpr: (ndarray) The false-positive rate.
    :param tpr: (ndarray) The true-positive rate.
    :param AUC: (float) The AUC(Area Under the Curve) of the ROC curve.
    :param title: (str or None) If a title is not None, print the title on graphs.
    :param label: (str or None) If a label is not None, set the label as a label of a line.
    :param color: ()
    :return:
    """
    if title is None:
        title = 'ROC curve'

    if label is None:
        label = 'Regression'

    plt.plot(fpr, tpr, color=color, label=label + ' (area = %0.2f)' % AUC)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")


# noinspection PyPep8Naming
def evaluate_predictions(y_actual: pd.Series, y_prediction: pd.Series,
                         title=None, confusion_matrix_plotting=False, roc_curve_plotting=False):
    """

    :param y_actual: (Series) The actual y values.
    :param y_prediction: (Series) The predicted y values.
    :param title: (str or None) If a title is not None, print the title on graphs.
    :param confusion_matrix_plotting: (bool) If confusion_matrix_plotting is True, plot the confusion matrix.
    :param roc_curve_plotting: (bool) If roc_curve_plotting is True,
        plot the ROC(Receiver Operating Characteristic) curve.

    :return accuracy: (float) The portion of correct predictions.
    :return f1_score: (float) The harmonic mean of precision and recall
    :return AUC: (float) The AUC(Area Under the Curve) of the ROC(Receiver Operating Characteristic) curve.
    """
    y_actual = y_actual.reset_index(drop=True)
    y_prediction = y_prediction.reset_index(drop=True)
    assert len(y_actual) == len(y_prediction)

    confusion_matrix = cm(y_actual, y_prediction)
    TN, FP, FN, TP = confusion_matrix.ravel()
    Precision = TP / (TP + FP + 1e-20)  # The portion of actual 1 of prediction 1.
    Recall = TP / (TP + FN + 1e-20)  # The portion of prediction 1 of actual 1.
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # The portion of correct predictions.
    f1_score = 2 / (1 / Precision + 1 / Recall + 1e-20)  # The harmonic mean of precision and recall.

    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(y_actual, y_prediction)
    AUC = auc(fpr, tpr)

    if confusion_matrix_plotting:
        plot_confusion_matrix(y_actual, y_prediction, title=title)

    if roc_curve_plotting:
        plt.figure()
        plot_roc_curve(fpr, tpr, AUC, title=title)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.show()

    return accuracy, f1_score, AUC


# noinspection PyPep8Naming
def plot_roc_curves(y_actual, y_predictions):
    """

    :param y_actual: (Series) The actual y values.
    :param y_predictions: (Map[Series]) The map of predicted y values.
    """
    cmaps = get_cmaps(len(y_predictions))

    plt.figure()
    y_actual = y_actual.reset_index(drop=True)
    keys = y_predictions.keys()
    for index, key in enumerate(keys):
        y_predictions[key] = y_predictions[key].reset_index(drop=True)
        assert len(y_actual) == len(y_predictions[key])

        # Calculate ROC and AUC
        fpr, tpr, _ = roc_curve(y_actual, y_predictions[key])
        AUC = auc(fpr, tpr)

        plot_roc_curve(fpr, tpr, AUC, title='Regression Comparison', label=key, color=cmaps(index))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')


# An usage example.
if __name__ == '__main__':
    from data.data_reader import get_training_data
    from data.data_combinator import get_full_combinations
    from stats.regression_calculator import get_ridge_regression, get_logistic_regression, get_lasso_regression, \
        get_linear_discriminant_analysis, get_quadratic_discriminant_analysis, get_naive_bayes, get_random_forest, \
        get_select_more_follower_count

    alpha = 0.062

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    original_x_val = x_val.copy()
    x_train = get_full_combinations(x_train)
    x_val = get_full_combinations(x_val)

    title = 'Select more follower count'
    print(title)
    y_prediction = get_select_more_follower_count(x_train, y_train, original_x_val)
    accuracy, f1_score, AUC = evaluate_predictions(y_val, y_prediction, title=title,
                                                   confusion_matrix_plotting=True,
                                                   roc_curve_plotting=True)
    print('accuracy:{}'.format(accuracy))
    print('f1_score:{}'.format(f1_score))
    print('AUC:{}'.format(AUC))
    print('-' * 70)

    y_predictions = {
        'Ridge regression, alpha={}'.format(alpha): get_ridge_regression(x_train, y_train, x_val, alpha, summary=True),
        'Logistic regression, alpha={}'.format(alpha): get_logistic_regression(x_train, y_train, x_val, alpha),
        'Lasso regression, alpha={}'.format(alpha): get_lasso_regression(x_train, y_train, x_val, alpha, summary=True),
        'LDA regression, alpha={}'.format(alpha): get_linear_discriminant_analysis(x_train, y_train, x_val, alpha),
        'QDA regression, alpha={}'.format(alpha): get_quadratic_discriminant_analysis(x_train, y_train, x_val, alpha),
        'NB regression, alpha={}'.format(alpha): get_naive_bayes(x_train, y_train, x_val, alpha),
        'RF regression, alpha={}'.format(alpha): get_random_forest(x_train, y_train, x_val, alpha),
        'Select more follower count': get_select_more_follower_count(x_train, y_train, original_x_val)
    }
    plot_roc_curves(y_val, y_predictions)
    plt.show()
