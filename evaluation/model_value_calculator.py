# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 3.
"""
import pandas as pd
import numpy as np

# The column names of following_count
A_FOLLOWER_COUNT = 'A_following_count'
B_FOLLOWER_COUNT = 'B_following_count'

MARGIN = 10
COST = 10


def get_influencer_follower_counts(test_data, y, y_val, column_name):
    """

    :param test_data: (DataFrame)
        columns A_follower_count    | (int)
                A_following_count   | (int)
                A_listed_count      | (int)
                A_mentions_received | (float)
                A_retweets_received | (float)
                A_mentions_sent     | (float)
                A_retweets_sent     | (float)
                A_posts             | (float)
                A_network_feature_1 | (int)
                A_network_feature_2 | (float)
                A_network_feature_3 | (float)
                B_follower_count    | (int)
                B_following_count   | (int)
                B_listed_count      | (int)
                B_mentions_received | (float)
                B_retweets_received | (float)
                B_mentions_sent     | (float)
                B_retweets_sent     | (float)
                B_posts             | (float)
                B_network_feature_1 | (int)
                B_network_feature_2 | (float)
                B_network_feature_3 | (float)
    :param y: (Series) The y of test set. y could be a y_test or a y_prediction.
    :param column_name: (str) The name of follower_count. This must be A_follower_count or B_follower_count.

    :return influencer_follower_counts: (Series) The follower counts of influencers.
        If a user is not an influence, the result is 0.
    """
    assert column_name == A_FOLLOWER_COUNT or column_name == B_FOLLOWER_COUNT
    assert len(test_data) == len(y)
    assert len(y_val) == len(y)
    y_copy = y.copy()

    follower_counts = test_data[column_name]

    # If column_name == A_follower_count, we will calculate the sum of follower counts when A is an influencer.
    # Else, we will calculate the sum of follower counts when B is an influencer.
    # 1 in y_prediction means that A is an influencer.
    # So we use original y_prediction to calculate the sum of follower counts when A is an influencer,
    # and we use (1 - y_prediction) to calculate the sum of follower counts when B is an influencer.
    if column_name == B_FOLLOWER_COUNT:
        y_copy = 1 - y_copy

    influencer_follower_counts = follower_counts * y_copy * y_val

    return influencer_follower_counts


def get_model_value(test_data, y, y_val, is_y_prediction):
    """

    :param test_data: (DataFrame)
        columns A_follower_count    | (int)
                A_following_count   | (int)
                A_listed_count      | (int)
                A_mentions_received | (float)
                A_retweets_received | (float)
                A_mentions_sent     | (float)
                A_retweets_sent     | (float)
                A_posts             | (float)
                A_network_feature_1 | (int)
                A_network_feature_2 | (float)
                A_network_feature_3 | (float)
                B_follower_count    | (int)
                B_following_count   | (int)
                B_listed_count      | (int)
                B_mentions_received | (float)
                B_retweets_received | (float)
                B_mentions_sent     | (float)
                B_retweets_sent     | (float)
                B_posts             | (float)
                B_network_feature_1 | (int)
                B_network_feature_2 | (float)
                B_network_feature_3 | (float)
    :param y: (Series) If is_y_prediction is True, this is the predictions of test set.
        Else, this is the true answer of test set.
    :param y_val: (Series) The true answer of test set.
    :param is_y_prediction: (bool) If is_y_prediction is True,
        use 0.00075 for chance that his/her followers will buy one unit of a product.
        Else, use 0.0005 for chance that his/her followers will buy one unit of a product.
    :return model_value: (float) The value of model.
    """
    assert len(test_data) == len(y)

    if is_y_prediction:
        chance = 0.00075
    else:
        chance = 0.0005

    A_influencer_follower_counts = get_influencer_follower_counts(test_data, y, y_val, A_FOLLOWER_COUNT)
    B_influencer_follower_counts = get_influencer_follower_counts(test_data, y, y_val, B_FOLLOWER_COUNT)
    influencer_follower_counts = A_influencer_follower_counts + B_influencer_follower_counts
    model_values = chance * MARGIN * influencer_follower_counts - COST
    model_value = np.sum(model_values)

    return model_value


if __name__ == '__main__':
    from data.data_reader import get_training_data
    from stats.regression_calculator import get_ridge_regression
    from evaluation.evaluator import custom_round

    alpha = 0.063

    x_train, y_train, x_val, y_val = get_training_data(validation=True)

    y_prediction = get_ridge_regression(x_train, y_train, x_val, alpha)
    y_prediction = y_prediction.apply(custom_round)

    print('Unanalysed:{}'.format(get_model_value(x_val, y_val, y_val, is_y_prediction=False)))
    print('Perfectly analysed:{}'.format(get_model_value(x_val, y_val, y_val, is_y_prediction=True)))
    print('Our model:{}'.format(get_model_value(x_val, y_prediction, y_val, is_y_prediction=True)))
    print('All A model:{}'.format(get_model_value(x_val, pd.Series([1 for i in range(len(y_val))]), y_val, is_y_prediction=True)))
    print('All B model:{}'.format(get_model_value(x_val, pd.Series([0 for i in range(len(y_val))]), y_val, is_y_prediction=True)))
