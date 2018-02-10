
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

if __name__ == '__main__':
    from data.data_reader import get_training_data
    from data.data_combinator import get_full_combinations

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    x_train = get_full_combinations(x_train)
    x_val = get_full_combinations(x_val)

    LDA = lda()
    LDA.fit(x_train, y_train)
    LDA_prob = LDA.predict_proba(x_val)
    LDA_prob

    QDA = qda()
    QDA.fit(x_train, y_train)
    QDA_prob = QDA.predict_proba(x_val)
    QDA_prob

    GNB = GaussianNB()
    GNB.fit(x_train, y_train)
    GaussianNB_prob = GNB.predict_proba(x_val)
    GaussianNB_prob

    # alpha = 1.0
    LOG = LogisticRegression()
    LOG.fit(x_train, y_train)
    # RIDGE = Ridge(alpha=alpha)
    # RIDGE.fit(x_train, y_train)
    # LASSO = Lasso(alpha=alpha)
    # LASSO.fit(x_train, y_train)

    LDA_y_score = LDA.fit(x_train, y_train).decision_function(x_val)
    QDA_y_score = QDA.fit(x_train, y_train).decision_function(x_val)
    GNB_y_score = GNB.fit(x_train, y_train).predict_proba(x_val)
    LOG_y_score = LOG.fit(x_train, y_train).decision_function(x_val)
    # RIDGE_y_score = RIDGE.fit(x_train, y_train).predict_proba(x_val)
    # LASSO_y_score = LASSO.fit(x_train, y_train).predict_proba(x_val)



    fpr_LDA, tpr_LDA, threashold_LDA = roc_curve(y_val, LDA_y_score)
    fpr_QDA, tpr_QDA, threashold_QDA = roc_curve(y_val, QDA_y_score)
    fpr_GNB, tpr_GNB, threashold_GNB = roc_curve(y_val, -GNB_y_score[:, 0])
    fpr_LOG, tpr_LOG, threashold_LOG = roc_curve(y_val, LOG_y_score)
    # fpr_RIDGE, tpr_RIDGE, threashold_RIDGE = roc_curve(y_val, RIDGE_y_score)
    # fpr_LASSO, tpr_LASSO, threashold_LASSO = roc_curve(y_val, LASSO_y_score)

    plt.figure()
    lw = 2
    plt.plot(fpr_LDA, tpr_LDA, color='yellow', label='LDA ROC curve')
    plt.plot(fpr_QDA, tpr_QDA, color='blue', label='QDA ROC curve')
    plt.plot(fpr_GNB, tpr_GNB, color='red', label='GNB ROC curve')
    plt.plot(fpr_LOG, tpr_LOG, color='green', label='LOG ROC curve')
    # plt.plot(fpr_RIDGE, tpr_RIDGE, color='blue', label='RIDGE ROC curve')
    # plt.plot(fpr_LASSO, tpr_LASSO, color='green', label='LASSO ROC curve')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
