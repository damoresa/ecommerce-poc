"""Regression using the DNNRegressor Estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import aws_ecommerce_poc_dnn

TRAINING_STEPS = 300000
EVALUATING_STEPS = 10000


def main(argv):
    """Builds, trains, and evaluates the model."""
    assert len(argv) == 1

    # Generate the DNNRegressor
    regressor = aws_ecommerce_poc_dnn.estimator_fn(None, None)

    # Train the model.
    print(' ## Training ')

    def train_input_fn():
        return aws_ecommerce_poc_dnn.train_input_fn('.', None)

    regressor.train(input_fn=train_input_fn, steps=TRAINING_STEPS)

    # Evaluate how the model performs on data it has not yet seen.
    print(' ## Evaluating ')

    def eval_input_fn():
        return aws_ecommerce_poc_dnn.eval_input_fn('.', None)

    eval_result = regressor.evaluate(input_fn=eval_input_fn, steps=EVALUATING_STEPS)

    # The evaluation returns a Python dictionary. The "average_loss" key holds the
    # Mean Squared Error (MSE).
    average_loss = eval_result["average_loss"]

    # Convert MSE to Root Mean Square Error (RMSE).
    print("\n" + 80 * "*")
    print("\nMSE error for the test set: {:.0f}".format(average_loss))
    print("RMS error for the test set: {:.0f}".format(average_loss ** 0.5))
    print("\n" + 80 * "*")

    # Export the model
    print(' ## Exporting ')

    def serving_fn():
        return aws_ecommerce_poc_dnn.serving_input_fn(None)

    regressor.export_savedmodel(
        export_dir_base='output',
        serving_input_receiver_fn=serving_fn)

    print()


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
