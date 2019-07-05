from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from constants import constants

import trainer.featurizer as featurizer


def build_estimator(config, hidden_units=None):
    features = featurizer.get_dnn_columns()
    classifier = tf.estimator.DNNClassifier(
        config=config,
        # All features except the LABEL
        feature_columns=features,
        hidden_units=hidden_units or [10, 10],
        n_classes=2,
    )
    return classifier
