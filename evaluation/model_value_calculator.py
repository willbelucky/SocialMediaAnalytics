# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 3.
"""
import numpy as np
import pandas as pd

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
    :param y: (Series) The y of the test set. y could be a y_test or a y_prediction.
    :param y_val: (Series) The true y of the test set.
    :param column_name: (str) The name of follower_count. This must be A_follower_count or B_follower_count.

    :return influencer_follower_counts: (Series) The follower counts of influencers.
        If a user is not an influence, the result is 0.
    """
    assert column_name == A_FOLLOWER_COUNT or column_name == B_FOLLOWER_COUNT
    assert len(test_data) == len(y)
    assert len(y_val) == len(y)
    y_copy = y.copy()

    follower_counts = test_data[column_name].reset_index(drop=True)

    # If column_name == A_follower_count, we will calculate the sum of follower counts when A is an influencer.
    # Else, we will calculate the sum of follower counts when B is an influencer.
    # 1 in y_prediction means that A is an influencer.
    # So we use original y_prediction to calculate the sum of follower counts when A is an influencer,
    # and we use (1 - y_prediction) to calculate the sum of follower counts when B is an influencer.
    if column_name == B_FOLLOWER_COUNT:
        y_copy = 1 - y_copy
        y_val = 1 - y_val

    influencer_follower_counts = follower_counts * y_copy * y_val

    return influencer_follower_counts


# noinspection PyPep8Naming
def get_model_value(test_data, y, y_val, is_y_prediction=True):
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
    from data.data_combinator import get_full_combinations
    from stats.regression_calculator import get_ridge_regression, get_lasso_regression, get_select_more_follower_count

    ridge_alpha = 0.062
    lasso_alpha = 0.002

    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    original_x_val = x_val.copy().reset_index(drop=True)
    x_train = get_full_combinations(x_train)
    x_val = get_full_combinations(x_val)
    y_val = y_val.reset_index(drop=True)

    ridge_regression_y_prediction = get_ridge_regression(x_train, y_train, x_val, 0.062)
    lasso_regression_y_prediction = get_lasso_regression(x_train, y_train, x_val, lasso_alpha)
    select_more_follower_count_y_prediction = get_select_more_follower_count(x_train, y_train, original_x_val)

    print('Unanalysed:{0:.3f}'.format(get_model_value(original_x_val, y_val, y_val, is_y_prediction=False)))
    print('Perfectly analysed:{0:.3f}'.format(get_model_value(original_x_val, y_val, y_val)))
    print('ridge_regression(alpha={0:.3f}) model:{1:.3f}'.format(
        ridge_alpha, get_model_value(original_x_val, ridge_regression_y_prediction, y_val)))
    print('lasso_regression(alpha={0:.3f}) model:{1:.3f}'.format(
        lasso_alpha, get_model_value(original_x_val, lasso_regression_y_prediction, y_val)))
    print('select_more_follow_follower_count model:{0:.3f}'.format(
        get_model_value(original_x_val, select_more_follower_count_y_prediction, y_val)))
    print('All A model:{0:.3f}'.format(
        get_model_value(original_x_val, pd.Series([1 for i in range(len(y_val))]), y_val)))
    print('All B model:{0:.3f}'.format(
        get_model_value(original_x_val, pd.Series([0 for i in range(len(y_val))]), y_val)))
