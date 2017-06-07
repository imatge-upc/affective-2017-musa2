"""
Gets TensorFlow version in order to decide which functions to use, since versions 0.12 and 1.0 introduce major breaking
changes to the API.
"""

import tensorflow as tf


def get_tf_version():
    if tf.__version__.startswith('1'):
        return "1"
    elif tf.__version__.startswith('0.12'):
        return "0.12"
    elif tf.__version__.startswith('0.11'):
        return "0.11"
    else:
        raise ValueError('Non supported TensorFlow version')
