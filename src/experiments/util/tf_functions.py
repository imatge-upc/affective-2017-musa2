"""
Wrap basic TensorFlow functions to abstract API selection from the main code
"""

import tensorflow as tf

from .tf_version import get_tf_version

TF_VERSION = get_tf_version()


def concat(values, concat_dim, name='concat'):
    if TF_VERSION == '0.11':
        return tf.concat(concat_dim, values, name)
    elif TF_VERSION == '0.12':  # Version '0.12.head' uses concat_v2
        return tf.concat_v2(values, concat_dim, name)
    else:
        return tf.concat(values, concat_dim, name)


def summary_writer(train_dir, graph):
    if TF_VERSION == '0.11':
        return tf.train.SummaryWriter(train_dir, graph)
    else:
        return tf.summary.FileWriter(train_dir, graph)


def scalar_summary(name, tensor):
    """ Create a scalar summary """
    if TF_VERSION == '0.11':
        return tf.scalar_summary(name, tensor)
    else:
        return tf.summary.scalar(name, tensor)


def histogram_summary(name, tensor):
    """ Create a histogram summary """
    if TF_VERSION == '0.11':
        return tf.histogram_summary(name, tensor)
    else:
        return tf.summary.histogram(name, tensor)


def merge_summary(summaries):
    if TF_VERSION == '0.11':
        return tf.merge_summary(summaries)
    else:
        return tf.summary.merge(summaries)


def merge_all_summaries():
    if TF_VERSION == '0.11':
        return tf.merge_all_summaries()
    else:
        return tf.summary.merge_all()
