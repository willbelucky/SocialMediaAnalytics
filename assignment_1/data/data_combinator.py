# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 24.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


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
    :return combined_x: 44(variables) columns * N rows
        columns sub_follower_count          | (int) A_follower_count - B_follower_count
                div_follower_count          | (float) A_follower_count / B_follower_count
                log_div_follower_count      | (float) log(A_follower_count / B_follower_count)
                root_div_follower_count     | (float) root(A_follower_count / B_follower_count)
                sub_following_count         | (int) A_following_count - B_following_count
                div_following_count         | (float) A_following_count / B_following_count
                log_div_following_count     | (float) log(A_following_count / B_following_count)
                root_div_following_count    | (float) root(A_following_count / B_following_count)
                sub_listed_count            | (int) A_listed_count - B_listed_count
                div_listed_count            | (float) A_listed_count / B_listed_count
                log_div_listed_count        | (float) log(A_listed_count / B_listed_count)
                root_div_listed_count       | (float) root(A_listed_count / B_listed_count)
                sub_mentions_received       | (float) A_mentions_received - B_mentions_received
                div_mentions_received       | (float) A_mentions_received / B_mentions_received
                log_div_mentions_received   | (float) log(A_mentions_received / B_mentions_received)
                root_div_mentions_received  | (float) root(A_mentions_received / B_mentions_received)
                sub_retweets_received       | (float) A_retweets_received - B_retweets_received
                div_retweets_received       | (float) A_retweets_received / B_retweets_received
                log_div_retweets_received   | (float) log(A_retweets_received / B_retweets_received)
                root_div_retweets_received  | (float) root(A_retweets_received / B_retweets_received)
                sub_mentions_sent           | (float) A_mentions_sent - B_mentions_sent
                div_mentions_sent           | (float) A_mentions_sent / B_mentions_sent
                log_div_mentions_sent       | (float) log(A_mentions_sent / B_mentions_sent)
                root_div_mentions_sent      | (float) root(A_mentions_sent / B_mentions_sent)
                sub_retweets_sent           | (float) A_retweets_sent - B_retweets_sent
                div_retweets_sent           | (float) A_retweets_sent / B_retweets_sent
                log_div_retweets_sent       | (float) log(A_retweets_sent / B_retweets_sent)
                root_div_retweets_sent      | (float) root(A_retweets_sent / B_retweets_sent)
                sub_posts                   | (float) A_posts - B_posts
                div_posts                   | (float) A_posts / B_posts
                log_div_posts               | (float) log(A_posts / B_posts)
                root_div_posts              | (float) root(A_posts / B_posts)
                sub_network_feature_1       | (int) A_network_feature_1 - B_network_feature_1
                div_network_feature_1       | (float) A_network_feature_1 / B_network_feature_1
                log_div_network_feature_1   | (float) log(A_network_feature_1 / B_network_feature_1)
                root_div_network_feature_1  | (float) root(A_network_feature_1 / B_network_feature_1)
                sub_network_feature_2       | (float) A_network_feature_2 - B_network_feature_2
                div_network_feature_2       | (float) A_network_feature_2 / B_network_feature_2
                log_div_network_feature_2   | (float) log(A_network_feature_2 / B_network_feature_2)
                root_div_network_feature_2  | (float) root(A_network_feature_2 / B_network_feature_2)
                sub_network_feature_3       | (float) A_network_feature_3 - B_network_feature_3
                div_network_feature_3       | (float) A_network_feature_3 / B_network_feature_3
                log_div_network_feature_3   | (float) log(A_network_feature_3 / B_network_feature_3)
                root_div_network_feature_3  | (float) root(A_network_feature_3 / B_network_feature_3)
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
        x_copy['log_div_' + column_name] = np.log(
            (x_copy.iloc[:, i] + 1e-20) / (x_copy.iloc[:, i + variable_number] + 1e-20))
        x_copy['root_div_' + column_name] = np.sqrt(x_copy.iloc[:, i] / (x_copy.iloc[:, i + variable_number] + 1e-20))

    # Use only combined columns.
    combined_x = x_copy.iloc[:, column_number:]
    combined_columns = combined_x.columns

    # Transform features by scaling each feature to a given range.
    scaler = MinMaxScaler(feature_range=(0, 1))
    combined_x = scaler.fit_transform(combined_x)
    combined_x = pd.DataFrame(data=combined_x, columns=combined_columns)

    return combined_x


def get_self_combinations(x: pd.DataFrame, combination_function=get_full_combinations, degree=2, interaction_only=False,
                          include_bias=False):
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
    :param combination_function: (function) The function getting combined data.
    :param degree: (int) The degree of the polynomial features. Default = 2.
    :param interaction_only: (bool) If true, only interaction features are produced:
        features that are products of at most degree distinct input features (so not x[1] ** 2, x[0] * x[2] ** 3, etc.).
    :param include_bias: (bool) If True (default), then include a bias column,
        the feature in which all polynomial powers are zero
        (i.e. a column of ones - acts as an intercept term in a linear model).
    :return combined_x: (DataFrame)
    """
    x_copy = x.copy()
    # noinspection PyTypeChecker
    x_copy = combination_function(x_copy)
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    combined_x = poly.fit_transform(x_copy)

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
    :return combined_x: 11(variables) columns * N rows
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
    :return combined_x: 11(variables) columns * N rows
        columns div_follower_count      | (float) A_follower_count / B_follower_count
                div_following_count     | (float) A_following_count / B_following_count
                div_listed_count        | (float) A_listed_count / B_listed_count
                div_mentions_received   | (float) A_mentions_received / B_mentions_received
                div_retweets_received   | (float) A_retweets_received / B_retweets_received
                div_mentions_sent       | (float) A_mentions_sent / B_mentions_sent
                div_retweets_sent       | (float) A_retweets_sent / B_retweets_sent
                div_posts               | (float) A_posts / B_posts
                div_network_feature_1   | (float) A_network_feature_1 / B_network_feature_1
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


# noinspection PyUnresolvedReferences
def get_log_div_combinations(x: pd.DataFrame):
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
    :return combined_x: 11(variables) columns * N rows
        columns log_div_follower_count      | (float) log(A_follower_count / B_follower_count)
                log_div_following_count     | (float) log(A_following_count / B_following_count)
                log_div_listed_count        | (float) log(A_listed_count / B_listed_count)
                log_div_mentions_received   | (float) log(A_mentions_received / B_mentions_received)
                log_div_retweets_received   | (float) log(A_retweets_received / B_retweets_received)
                log_div_mentions_sent       | (float) log(A_mentions_sent / B_mentions_sent)
                log_div_retweets_sent       | (float) log(A_retweets_sent / B_retweets_sent)
                log_div_posts               | (float) log(A_posts / B_posts)
                log_div_network_feature_1   | (float) log(A_network_feature_1 / B_network_feature_1)
                log_div_network_feature_2   | (float) log(A_network_feature_2 / B_network_feature_2)
                log_div_network_feature_3   | (float) log(A_network_feature_3 / B_network_feature_3)
    """
    x_copy = x.copy()
    column_number = len(x_copy.columns)
    assert column_number % 2 == 0
    variable_number = column_number // 2

    columns = x_copy.columns
    for i in range(variable_number):
        column_name = columns[i][2:]
        # For preventing DivideByZeroError, add small number to the divisor.
        x_copy['log_div_' + column_name] = np.log(
            (x_copy.iloc[:, i] + 1e-20) / (x_copy.iloc[:, i + variable_number] + 1e-20))

    # Use only combined columns.
    combined_x = x_copy.iloc[:, column_number:]
    combined_columns = combined_x.columns

    # Transform features by scaling each feature to a given range.
    scaler = MinMaxScaler(feature_range=(0, 1))
    combined_x = scaler.fit_transform(combined_x)
    combined_x = pd.DataFrame(data=combined_x, columns=combined_columns)

    return combined_x


# noinspection PyUnresolvedReferences
def get_root_div_combinations(x: pd.DataFrame):
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
    :return combined_x: 11(variables) columns * N rows
        columns root_div_follower_count     | (float) root(A_follower_count / B_follower_count)
                root_div_following_count    | (float) root(A_following_count / B_following_count)
                root_div_listed_count       | (float) root(A_listed_count / B_listed_count)
                root_div_mentions_received  | (float) root(A_mentions_received / B_mentions_received)
                root_div_retweets_received  | (float) root(A_retweets_received / B_retweets_received)
                root_div_mentions_sent      | (float) root(A_mentions_sent / B_mentions_sent)
                root_div_retweets_sent      | (float) root(A_retweets_sent / B_retweets_sent)
                root_div_posts              | (float) root(A_posts / B_posts)
                root_div_network_feature_1  | (float) root(A_network_feature_1 / B_network_feature_1)
                root_div_network_feature_2  | (float) root(A_network_feature_2 / B_network_feature_2)
                root_div_network_feature_3  | (float) root(A_network_feature_3 / B_network_feature_3)
    """
    x_copy = x.copy()
    column_number = len(x_copy.columns)
    assert column_number % 2 == 0
    variable_number = column_number // 2

    columns = x_copy.columns
    for i in range(variable_number):
        column_name = columns[i][2:]
        # For preventing DivideByZeroError, add small number to the divisor.
        x_copy['root_div_' + column_name] = np.sqrt(x_copy.iloc[:, i] / (x_copy.iloc[:, i + variable_number] + 1e-20))

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
