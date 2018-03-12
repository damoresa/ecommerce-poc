import numpy as np
import os
import tensorflow as tf
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn

def estimator_fn(run_config, hyperparameters):
    # Defines the features columns that will be the input of the estimator
    feature_columns = [
        tf.feature_column.numeric_column(key="input", shape=4),
    ]
    # Returns the instance of estimator.
    return tf.estimator.DNNRegressor(hidden_units=[50, 25], feature_columns=feature_columns, config=run_config)


def train_input_fn(training_dir, hyperparameters):
    # invokes _input_fn with training dataset
    return _input_fn(training_dir, 'AWS-Ecommerce-Train.csv')


def eval_input_fn(training_dir, hyperparameters):
    # invokes _input_fn with evaluation dataset
    return _input_fn(training_dir, 'AWS-Ecommerce-Test.csv')


def _input_fn(training_dir, training_filename):
    # reads the dataset using tf.dataset API
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=os.path.join(training_dir, training_filename), target_dtype=np.float32, features_dtype=np.float32)

    # returns features x and labels y
    return tf.estimator.inputs.numpy_input_fn(
        x={"input": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)()


def serving_input_fn(hyperparameters):
    # defines the input placeholder
    avg_session = tf.placeholder(tf.float32)
    app_time = tf.placeholder(tf.float32)
    web_time = tf.placeholder(tf.float32)
    membership = tf.placeholder(tf.float32)
    # returns the ServingInputReceiver object.
    return build_raw_serving_input_receiver_fn({"input": np.array(avg_session, app_time, web_time, membership)})()