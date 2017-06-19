#  -*- coding: utf-8 -*-
"""
Predict Housing price in Boston, pratice on using high-level API
Algorithms: Neutral Network
Reference: https://www.tensorflow.org/get_started/tflearn

Date: Jun 14, 2017
@author: Thuong Tran
@Library: tensorflow - high-level API with tf.contrib.learn
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_PATH = "./tmp/data/boston_train.csv"
TEST_PATH = "./tmp/data/boston_test.csv"
PRED_PATH = "./tmp/data/boston_predict.csv"
MODEL_DIR = "./tmp/boston_models"

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"

training_set = pd.read_csv(TRAIN_PATH, skipinitialspace = True,
                          skiprows = 1, names = COLUMNS)
test_set = pd.read_csv(TEST_PATH, skipinitialspace = True,
                      skiprows = 1, names = COLUMNS)
prediction_set = pd.read_csv(PRED_PATH, skipinitialspace = True,
                            skiprows = 1, names = COLUMNS)

feature_columns = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

regressor = tf.contrib.learn.DNNRegressor(feature_columns = feature_columns,
                                          hidden_units = [10, 10],
                                          model_dir = MODEL_DIR)
def input_fn(data_set):
  feature_columns = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_columns, labels

regressor.fit(input_fn = lambda: input_fn(training_set), steps = 5000)

ev = regressor.evaluate(input_fn = lambda: input_fn(test_set), steps = 1)
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

y = regressor.predict(input_fn = lambda: input_fn(prediction_set))
# .predict() returns an iterator; convert to a list and print predictions
predictions = list(itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))



