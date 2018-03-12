import numpy as np
import os
import tensorflow as tf

def estimator_fn(run_config, hyperparameters):
    # Defines the features columns that will be the input of the estimator
    feature_columns = [
        tf.feature_column.numeric_column(key="input", shape=[4]),
    ]
    # Returns the instance of estimator.
    return tf.estimator.DNNRegressor(hidden_units=[50, 25], feature_columns=feature_columns, config=run_config)


def train_input_fn(training_dir, hyperparameters):
    # invokes _input_fn with training dataset
    print(' # train_input_fn ')
    return _input_fn(training_dir, 'AWS-Ecommerce-Train.csv')


def eval_input_fn(training_dir, hyperparameters):
    # invokes _input_fn with evaluation dataset
    print(' # eval_input_fn ')
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
    # FIXME: FixedLenFeature' object has no attribute 'get_shape'
    feature_spec = {'input': tf.FixedLenFeature(dtype=tf.float32, shape=[4])}
    # returns the ServingInputReceiver object.
    return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)()