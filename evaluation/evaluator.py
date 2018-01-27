# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import roc_curve, auc


# noinspection PyPep8Naming
def plot_roc_curve(fpr, tpr, AUC, title=None):
    """

    :param fpr: (ndarray) The false-positive rate.
    :param tpr: (ndarray) The true-positive rate.
    :param AUC: (float) The AUC(Area Under the Curve) of the ROC curve.
    :param title: (str or None) If a title is not None, print the title on graphs.
    :return:
    """
    if title is None:
        title = 'ROC curve'

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def custom_round(number):
    return 1 if number >= 0.5 else 0


# noinspection PyPep8Naming
def evaluate_predictions(y_actual, y_prediction,
                         title=None,
                         confusion_matrix_plotting=False,
                         roc_curve_plotting=False):
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

    y_prediction_round = y_prediction.apply(custom_round)
    confusion_matrix = cm(y_actual, y_prediction_round)
    TN, FP, FN, TP = confusion_matrix.ravel()
    Precision = TP / (TP + FP + 1e-20)  # The portion of actual 1 of prediction 1.
    Recall = TP / (TP + FN + 1e-20)  # The portion of prediction 1 of actual 1.
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # The portion of correct predictions.
    f1_score = 2 / (1 / Precision + 1 / Recall + 1e-20)  # The harmonic mean of precision and recall.

    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(y_actual, y_prediction_round)
    AUC = auc(fpr, tpr)

    if confusion_matrix_plotting:
        plot_confusion_matrix(y_actual, y_prediction_round, title=title)

    if roc_curve_plotting:
        plot_roc_curve(fpr, tpr, AUC, title=title)
        plt.show()

    return accuracy, f1_score, AUC


# An usage example.
if __name__ == '__main__':
    from data.data_reader import get_training_data
    from data.data_combinator import get_full_combinations
    from stats.regression_calculator import get_lasso_regression

    alpha = 0.002

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    x_train = get_full_combinations(x_train)
    x_val = get_full_combinations(x_val)

    title = 'Lasso regression, alpha={}'.format(alpha)
    print(title)
    y_prediction = get_lasso_regression(x_train, y_train, x_val, alpha)
    accuracy, f1_score, AUC = evaluate_predictions(y_val, y_prediction, title=title,
                                                   confusion_matrix_plotting=True,
                                                   roc_curve_plotting=True)
    print('accuracy:{}'.format(accuracy))
    print('f1_score:{}'.format(f1_score))
    print('AUC:{}'.format(AUC))
    print('-' * 70)
