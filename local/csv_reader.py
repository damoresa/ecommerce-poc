from __future__ import absolute_import
from __future__ import print_function

import collections
import tensorflow as tf

defaults = collections.OrderedDict([
    ("email", ['asd@asd.as']),
    ("avatar", ['DarkGreen']),
    ("avg-session", [0.0]),
    ("app-time", [0.0]),
    ("web-time", [0.0]),
    ("membership", [0.0]),
    ("yearly-amount", [0.0]),
])

nonFeature = collections.OrderedDict([
    ("email", ['asd@asd.as']),
    ("avatar", ['DarkGreen']),
])


def dataset(filepath, label_name="yearly-amount", train_fraction=0.7):

    def decode_line(line):
        """Convert a csv line into a (features_dict,label) pair."""
        items = tf.decode_csv(line, list(defaults.values()))

        # Convert the keys and items to a dict.
        pairs = zip(defaults.keys(), items)
        features_dict = dict(pairs)

        # Remove the label from the features_dict
        label = features_dict.pop(label_name)

        # Remove the non-feature columns
        for key in nonFeature:
            features_dict.pop(key)

        return features_dict, label

    def in_training_set(line):
        """Returns a boolean tensor, true if the line is in the training set."""
        # If you randomly split the dataset you won't get the same split in both
        # sessions if you stop and restart training later. Also a simple
        # random split won't work with a dataset that's too big to `.cache()` as
        # we are doing here.
        num_buckets = 1000000
        bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
        # Use the hash bucket id as a random number that's deterministic per example
        return bucket_id < int(train_fraction * num_buckets)

    def in_test_set(line):
        """Returns a boolean tensor, true if the line is in the training set."""
        # Items not in the training set are in the test set.
        # This line must use `~` instead of `not` because `not` only works on python
        # booleans but we are dealing with symbolic tensors.
        return ~in_training_set(line)

    base_dataset = (
        tf.data
            # Get the lines from the file.
            .TextLineDataset(filepath)
            # Skip the header
            .skip(1))

    train = (base_dataset
             # Take only the training-set lines.
             .filter(in_training_set)
             # Decode each line into a (features_dict, label) pair.
             .map(decode_line)
             # Cache data so you only decode the file once.
             .cache())

    # Do the same for the test-set.
    test = (base_dataset.filter(in_test_set).cache().map(decode_line))

    return train, test

