# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# noinspection PyUnresolvedReferences
def get_full_combinations(x: pd.DataFrame):
    """

    :param x: (DataFrame) 22(variables) columns * N rows
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
    :return combined_x: 22(variables) columns * N rows
        columns sub_follower_count      | (int) A_follower_count - B_follower_count
                div_follower_count      | (int) A_follower_count / B_follower_count
                sub_following_count     | (int) A_following_count - B_following_count
                div_following_count     | (int) A_following_count / B_following_count
                sub_listed_count        | (int) A_listed_count - B_listed_count
                div_listed_count        | (int) A_listed_count / B_listed_count
                sub_mentions_received   | (float) A_mentions_received - B_mentions_received
                div_mentions_received   | (float) A_mentions_received / B_mentions_received
                sub_retweets_received   | (float) A_retweets_received - B_retweets_received
                div_retweets_received   | (float) A_retweets_received / B_retweets_received
                sub_mentions_sent       | (float) A_mentions_sent - B_mentions_sent
                div_mentions_sent       | (float) A_mentions_sent / B_mentions_sent
                sub_retweets_sent       | (float) A_retweets_sent - B_retweets_sent
                div_retweets_sent       | (float) A_retweets_sent / B_retweets_sent
                sub_posts               | (float) A_posts - B_posts
                div_posts               | (float) A_posts / B_posts
                sub_network_feature_1   | (int) A_network_feature_1 - B_network_feature_1
                div_network_feature_1   | (int) A_network_feature_1 / B_network_feature_1
                sub_network_feature_2   | (float) A_network_feature_2 - B_network_feature_2
                div_network_feature_2   | (float) A_network_feature_2 / B_network_feature_2
                sub_network_feature_3   | (float) A_network_feature_3 - B_network_feature_3
                div_network_feature_3   | (float) A_network_feature_3 / B_network_feature_3
    """
    x_copy = x.copy()
    column_number = len(x_copy.columns)
    assert column_number % 2 == 0
    variable_number = column_number // 2

    columns = x_copy.columns
    for i in range(variable_number):
        column_name = columns[i][2:]
        x_copy['sub_' + column_name] = x_copy.iloc[:, i] - x_copy.iloc[:, i + variable_number]
        # For preventing DivideByZeroError, add small number to the divisor.
        x_copy['div_' + column_name] = x_copy.iloc[:, i] / (x_copy.iloc[:, i + variable_number] + 1e-20)

    # Use only combined columns.
    combined_x = x_copy.iloc[:, column_number:]
    combined_columns = combined_x.columns

    # Transform features by scaling each feature to a given range.
    scaler = MinMaxScaler(feature_range=(0, 1))
    combined_x = scaler.fit_transform(combined_x)
    combined_x = pd.DataFrame(data=combined_x, columns=combined_columns)

    return combined_x


# noinspection PyUnresolvedReferences
def get_sub_combinations(x: pd.DataFrame):
    """

    :param x: (DataFrame) 22(variables) columns * N rows
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
    :return combined_x: 22(variables) columns * N rows
        columns sub_follower_count      | (int) A_follower_count - B_follower_count
                sub_following_count     | (int) A_following_count - B_following_count
                sub_listed_count        | (int) A_listed_count - B_listed_count
                sub_mentions_received   | (float) A_mentions_received - B_mentions_received
                sub_retweets_received   | (float) A_retweets_received - B_retweets_received
                sub_mentions_sent       | (float) A_mentions_sent - B_mentions_sent
                sub_retweets_sent       | (float) A_retweets_sent - B_retweets_sent
                sub_posts               | (float) A_posts - B_posts
                sub_network_feature_1   | (int) A_network_feature_1 - B_network_feature_1
                sub_network_feature_2   | (float) A_network_feature_2 - B_network_feature_2
                sub_network_feature_3   | (float) A_network_feature_3 - B_network_feature_3
    """
    x_copy = x.copy()
    column_number = len(x_copy.columns)
    assert column_number % 2 == 0
    variable_number = column_number // 2

    columns = x_copy.columns
    for i in range(variable_number):
        column_name = columns[i][2:]
        x_copy['sub_' + column_name] = x_copy.iloc[:, i] - x_copy.iloc[:, i + variable_number]

    # Use only combined columns.
    combined_x = x_copy.iloc[:, column_number:]
    combined_columns = combined_x.columns

    # Transform features by scaling each feature to a given range.
    scaler = MinMaxScaler(feature_range=(0, 1))
    combined_x = scaler.fit_transform(combined_x)
    combined_x = pd.DataFrame(data=combined_x, columns=combined_columns)

    return combined_x


# noinspection PyUnresolvedReferences
def get_div_combinations(x: pd.DataFrame):
    """

    :param x: (DataFrame) 22(variables) columns * N rows
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
    :return combined_x: 22(variables) columns * N rows
        columns div_follower_count      | (int) A_follower_count / B_follower_count
                div_following_count     | (int) A_following_count / B_following_count
                div_listed_count        | (int) A_listed_count / B_listed_count
                div_mentions_received   | (float) A_mentions_received / B_mentions_received
                div_retweets_received   | (float) A_retweets_received / B_retweets_received
                div_mentions_sent       | (float) A_mentions_sent / B_mentions_sent
                div_retweets_sent       | (float) A_retweets_sent / B_retweets_sent
                div_posts               | (float) A_posts / B_posts
                div_network_feature_1   | (int) A_network_feature_1 / B_network_feature_1
                div_network_feature_2   | (float) A_network_feature_2 / B_network_feature_2
                div_network_feature_3   | (float) A_network_feature_3 / B_network_feature_3
    """
    x_copy = x.copy()
    column_number = len(x_copy.columns)
    assert column_number % 2 == 0
    variable_number = column_number // 2

    columns = x_copy.columns
    for i in range(variable_number):
        column_name = columns[i][2:]
        # For preventing DivideByZeroError, add small number to the divisor.
        x_copy['div_' + column_name] = x_copy.iloc[:, i] / (x_copy.iloc[:, i + variable_number] + 1e-20)

    # Use only combined columns.
    combined_x = x_copy.iloc[:, column_number:]
    combined_columns = combined_x.columns

    # Transform features by scaling each feature to a given range.
    scaler = MinMaxScaler(feature_range=(0, 1))
    combined_x = scaler.fit_transform(combined_x)
    combined_x = pd.DataFrame(data=combined_x, columns=combined_columns)

    return combined_x


# An usage example
if __name__ == '__main__':
    from data.data_reader import get_training_data

    x_train, y_train, _, _ = get_training_data()
    full_combined_x_train = get_full_combinations(x_train)
    print(full_combined_x_train.head())
    print('-' * 70)

    sub_combined_x_train = get_sub_combinations(x_train)
    print(sub_combined_x_train.head())
    print('-' * 70)

    div_combined_x_train = get_div_combinations(x_train)
    print(div_combined_x_train.head())
    print('-' * 70)
