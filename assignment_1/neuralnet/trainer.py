# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 27.
"""
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

from data.data_combinator import get_full_combinations
from data.data_reader import get_training_data
from util.machine_learning_supporter import SupervisedData


def get_data(job_name):
    # the data, shuffled and split between train and test sets
    x_train, y_train, x_val, y_val = get_training_data(validation=True)
    x_train = get_full_combinations(x_train)
    x_val = get_full_combinations(x_val)
    y_val = y_val.reset_index(drop=True)

    input_shape = (len(x_train.columns),)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'test samples')

    return SupervisedData(job_name, x_train, y_train, x_val, y_val, input_shape)


def get_model(input_shape):
    model = Sequential()
    model.add(Dense(1024, activation=K.relu, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation=K.relu))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation=K.relu))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation=K.tanh))

    model.compile(
        loss=K.binary_crossentropy,
        optimizer=Adam(),
        metrics=['accuracy']
    )

    model.summary()

    return model


if __name__ == '__main__':
    job_name = 'influencer_prediction'

    batch_size = 128
    epochs = 20

    data = get_data(job_name)

    model = get_model(data.input_shape)

    data.evaluate_model(model, batch_size, epochs)
