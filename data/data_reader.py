# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 23.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

TRAINING_DOWNLOAD_URL = 'https://www.dropbox.com/s/newxt7ifuipiezp/train.csv?dl=1'
TEST_DOWNLOAD_URL = 'https://www.dropbox.com/s/dhqm40csvi0mhhz/test.csv?dl=1'
TARGET = 'Choice'


def get_training_data(validation: bool=False, validation_size: float=0.2) \
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame or None, pd.DataFrame or None):
    """
    (1(target: 'Choice') + 22(variables)) columns * 5500 rows

    :param validation: (bool) If validation is True, split the train set to train set and validation set
                                 and return them.
    :param validation_size: (float) The portion of validation set.

    :return x_train: (DataFrame) 22(variables) columns * (5500 * (1 - validation_size)) rows
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
    :return y_train: (Series[int]) (target: 'Choice') * (5500 * (1 - validation_size))
    :return x_val: (DataFrame) 22(variables) columns * (5500 * validation_size) rows
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
    :return y_val: (Series[int]) (target: 'Choice') * (5500 * validation_size)
    """
    if validation and (validation_size <= 0 or validation_size >= 1):
        raise ValueError('validation_size should be bigger than 0 and smaller than 1.')

    training_dataframe = pd.read_csv(TRAINING_DOWNLOAD_URL)
    x_train = training_dataframe.loc[:, training_dataframe.columns != TARGET]
    y_train = training_dataframe.loc[:, TARGET]
    x_val = None
    y_val = None

    if validation:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, random_state=1)

    return x_train, y_train, x_val, y_val


def get_test_data():
    """
    22(variables) columns * 5952 rows

    :return x_test: (DataFrame) 22(variables) columns * 5500 rows
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
    """
    x_test = pd.read_csv(TEST_DOWNLOAD_URL)
    return x_test


# Usage examples
if __name__ == '__main__':
    # Do not use validation set.
    x_train, y_train, _, _ = get_training_data()
    print('len(x_train) is 5500.')
    print(x_train.head())
    print('-' * 70)
    print('len(y_train) is 5500.')
    print(y_train.head())
    print('-' * 70)

    # Use validation set.
    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    print('len(x_train) is (5500 * (1 - 0.2)) = 4400.')
    print(x_train.head())
    print('-' * 70)
    print('len(y_train) is (5500 * (1 - 0.2)) = 4400.')
    print(y_train.head())
    print('-' * 70)
    print('len(x_val) is (5500 * 0.2) = 1100.')
    print(x_val.head())
    print('-' * 70)
    print('len(y_val) is (5500 * 0.2) = 1100.')
    print(y_val.head())
    print('-' * 70)

    # Use test set.
    x_test = get_test_data()
    print('len(x_test) is 5952.')
    print(x_test.head())
    print('-' * 70)
