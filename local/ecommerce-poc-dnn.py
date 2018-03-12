"""Regression using the DNNRegressor Estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from local import csv_reader

STEPS = 300000

def main(argv):
    """Builds, trains, and evaluates the model."""
    assert len(argv) == 1
    (train, test) = csv_reader.dataset('Clean-Ecommerce.csv')

    # Build the training input_fn.
    def input_train():
        return (
            # Shuffling with a buffer larger than the data set ensures
            # that the examples are well mixed.
            train.shuffle(1000).batch(128)
                # Repeat forever
                .repeat().make_one_shot_iterator().get_next())

    # Build the validation input_fn.
    def input_test():
        return (test.shuffle(1000).batch(128)
                .make_one_shot_iterator().get_next())

    feature_columns = [
        tf.feature_column.numeric_column(key="avg-session"),
        tf.feature_column.numeric_column(key="app-time"),
        tf.feature_column.numeric_column(key="web-time"),
        tf.feature_column.numeric_column(key="membership"),
    ]

    # Build a DNNRegressor, with 2x20-unit hidden layers, with the feature columns
    # defined above as input.
    model = tf.estimator.DNNRegressor(
        hidden_units=[50, 25], feature_columns=feature_columns)

    # Train the model.
    model.train(input_fn=input_train, steps=STEPS)

    # Evaluate how the model performs on data it has not yet seen.
    eval_result = model.evaluate(input_fn=input_test)

    # The evaluation returns a Python dictionary. The "average_loss" key holds the
    # Mean Squared Error (MSE).
    average_loss = eval_result["average_loss"]

    # Convert MSE to Root Mean Square Error (RMSE).
    print("\n" + 80 * "*")
    print("\nRMS error for the test set: {:.0f}".format(average_loss**0.5))

    # Run the model in prediction mode.
    input_dict = {
        "avg-session": np.array([34.49726772511229]),
        "app-time": np.array([12.65565114916675]),
        "web-time": np.array([39.57766801952616]),
        "membership": np.array([4.0826206329529615])
    }
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        input_dict, shuffle=False)
    predict_results = model.predict(input_fn=predict_input_fn)

    # Print the prediction results.
    print("\nPrediction results:")
    for i, prediction in enumerate(predict_results):
        msg = ("Average session: {: 2.14f}, "
               "Application time: {: 2.14f}, "
               "Web time: {: 2.14f}, "
               "Membership: {: 2.16f}, "
               "Prediction: {: 10.14f}")
        msg = msg.format(input_dict["avg-session"][i], input_dict["app-time"][i], input_dict["web-time"][i],
                         input_dict["membership"][i], prediction["predictions"][0])

        print("    " + msg)

    print()


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
tf.app.run(main=main)