from __future__ import absolute_import

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as pretrained_nets


def resnet_v1_50(inputs, num_classes, scope=None, reuse=None, is_training=False, weight_decay_rate=0.0001,
                 batch_norm_decay=0.997, batch_norm_epsilon=1e-5, batch_norm_scale=True, fcn=False):
    resnet_v1 = pretrained_nets.resnet_v1
    with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=is_training,
                                                   weight_decay=weight_decay_rate,
                                                   batch_norm_decay=batch_norm_decay,
                                                   batch_norm_epsilon=batch_norm_epsilon,
                                                   batch_norm_scale=batch_norm_scale)):
        logits, end_points = resnet_v1.resnet_v1_50(inputs,
                                                    num_classes=num_classes,
                                                    global_pool=True,
                                                    output_stride=None,
                                                    reuse=reuse,
                                                    scope=scope)
        if not fcn:
            logits = tf.reduce_mean(logits, [1, 2])
    return logits, end_points


def fully_connected_layer(input_tensor, num_output_units, weight_decay, name):
    num_input_units = input_tensor.get_shape()[1]

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    weights = tf.get_variable('%s/weights' % name,
                              shape=[num_input_units, num_output_units],
                              initializer=tf.truncated_normal_initializer(stddev=0.01),
                              dtype=tf.float32,
                              regularizer=regularizer,
                              trainable=True)
    biases = tf.get_variable('%s/bias' % name,
                             shape=[num_output_units],
                             initializer=tf.zeros_initializer,
                             dtype=tf.float32,
                             trainable=True)
    fc = tf.nn.xw_plus_b(input_tensor, weights, biases)
    return fc


def projection_matrix(input_tensor, num_output_units, weight_decay, name):
    num_input_units = input_tensor.get_shape()[1].value

    input_tensor = tf.reshape(input_tensor, [-1, num_input_units])

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    weights = tf.get_variable('%s/weights' % name,
                              shape=[num_input_units, num_output_units],
                              initializer=tf.truncated_normal_initializer(stddev=0.01),
                              dtype=tf.float32,
                              regularizer=regularizer,
                              trainable=True)
    fc = tf.matmul(input_tensor, weights)
    return fc


def affect_classifier(logits_anp, weight_decay_rate, n_units, task='emotion'):
    """
    Generates NN to perform emotion classification
    :param logits_anp: output of the CNN
    :param weight_decay_rate: weight decay rate
    :param n_units: list of integers with the number of units for each layer. len(n_units) layers will be created.
    :param task: prefix for the created layers, e.g. task_hidden, task_classif
    :return:
    """
    if not isinstance(n_units, list):
        n_units = [n_units]

    net = logits_anp
    for i, n in enumerate(n_units[0:-1]):
        with tf.variable_scope('%s_hidden' % task):
            net = fully_connected_layer(net, n, weight_decay_rate, 'fc%d' % (i+1))
            net = tf.nn.relu(net, name='relu%d' % (i+1))

    with tf.variable_scope('%s_linear' % task):
        out = fully_connected_layer(net, n_units[-1], weight_decay_rate, 'classif')

    return out
