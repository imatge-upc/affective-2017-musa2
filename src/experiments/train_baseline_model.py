"""
Train softmax baseline on MVSO_EN_1200 in a multi-GPU environment.

This code is based on the Inception tutorial in the tensorflow/models repository.

Usage example:

    python 02_train_softmax_model.py \
        --data_dir path_to_mvso_train_data_dir \
        --train_dir path_to_output_dir \
        --batch_size 32 \
        --image_size 224 \
        --max_steps 1000000 \
        --num_gpus 4 \
        --cnn resnet50 \
        --optimizer rmsprop \
        --initial_learning_rate 0.1 \
        --weight_decay_rate 0.0001 \
        --remove_dir \
        --checkpoint path_to_model/resnet_v1_50_anp.ckpt

"""


from __future__ import absolute_import

import time
import datetime
import numpy as np
import tensorflow as tf

from util.training import *
import util.tf_functions as tf_
from util.nn import resnet_v1_50
from util.mtvso_data import MTVSOData
from util import batch_generator_mvso
from util.vgg_preprocessing import preprocess_image

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS


def tower_loss(scope, reuse):
    """
    Calculate the total loss on a single tower.
    Args:
      scope: unique prefix string identifying the tower, e.g. 'tower_0'
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # Get images and labels
    dataset = MTVSOData(subset='train')
    images, labels = batch_generator_mvso.generate_batch(dataset,
                                                         FLAGS.batch_size,
                                                         train=True,
                                                         image_processing_fn=preprocess_image)

    # Build inference Graph.
    if FLAGS.cnn == 'resnet50':
        tf.logging.info("Using label in resnet: " + str(int(FLAGS.label)))
        logits, _ = resnet_v1_50(images, dataset.num_classes()[int(FLAGS.label)],
                                 scope='resnet_v1_50',
                                 reuse=reuse,
                                 is_training=True,
                                 weight_decay_rate=FLAGS.weight_decay_rate,
                                 batch_norm_decay=0.997,
                                 batch_norm_epsilon=1e-5,
                                 batch_norm_scale=True)

    else:
        raise ValueError('The specified CNN architecture is not supported')

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = cross_entropy_loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        loss_name = l.op.name
        tf.logging.info('Creating summary for %s', l.op.name)
        # Name each loss as '_raw' and name the moving average version of the loss
        # as the original loss name.
        tf_.scalar_summary(loss_name + '_raw', l)
        tf_.scalar_summary(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss, logits, labels


def train():
    """Train for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Decay the learning rate exponentially based on the number of steps.
        lr = create_learning_rate_scheduler(global_step, dataset=MTVSOData(subset='train'))

        # Create an optimizer that performs gradient descent.
        opt = create_optimizer(lr)

        # Calculate the gradients for each model tower.
        tower_grads, tower_logits, tower_labels, tower_losses = [], [], [], []
        reuse = None
        # tf.variable_scope outside the loop is needed for the code to work on TensorFlow versions >=0.12
        # https://github.com/tensorflow/tensorflow/issues/6220#issuecomment-266425068
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        # Calculate the loss for one tower. This function constructs
                        # the entire model but shares the variables across all towers.
                        loss, logits, labels = tower_loss(scope, reuse)

                        # Reuse variables for the next tower.
                        reuse = True
                        #tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
                        tower_logits.append(logits)
                        tower_labels.append(labels)
                        tower_losses.append(loss)

        # Concatenate the outputs of all towers
        logits_op = concat(tower_logits, 0, 'concat_logits')
        labels_op = concat(tower_labels, 0, 'concat_labels')
        loss_op = tf.reduce_mean(tower_losses)

        # Update BN's moving_mean and moving_variance
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            tf.logging.info('Gathering update_ops')
            with tf.control_dependencies(tf.tuple(update_ops)):
                loss_op = tf.identity(loss_op)

        # Track the loss of all towers
        summaries.append(tf_.scalar_summary('combined_loss', loss_op))

        # Compute top-1 accuracy
        top1_accuracy_op = top_k_accuracy(logits_op, labels_op, k=1)

        # Compute top-5 accuracy
        top5_accuracy_op = top_k_accuracy(logits_op, labels_op, k=5)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf_.scalar_summary('learning_rate', lr))

        # Add histograms for trainable variables and gradients.
        maybe_track_vars_and_gradients(grads, summaries)

        for op in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            tf.logging.info(op.name)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU implementations.
        sess = tf.InteractiveSession(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Restore model weights
        restore_model(sess, saver, current_scope='resnet_v1_50')

        # Manually set the learning rate if there is no learning rate decay and we are resuming training
        overwrite_learning_rate(sess, lr)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf_.summary_writer(FLAGS.train_dir, sess.graph)
        accumulated_top1_accuracy_10_steps, accumulated_top1_accuracy_100_steps = 0., 0.
        accumulated_top5_accuracy_10_steps, accumulated_top5_accuracy_100_steps = 0., 0.

        saver.save(sess, "/work/awoodward/mtvso/mtvso_out/experiment_2/model/model.ckpt", write_meta_graph=False, write_state=False)

        for step in range(FLAGS.max_steps):
            g_step = global_step.eval()
            start_time = time.time()
            _, loss_value, top1_accuracy_value, top5_accuracy_value = sess.run([train_op, loss_op,
                                                                                top1_accuracy_op,
                                                                                top5_accuracy_op])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            accumulated_top1_accuracy_10_steps += top1_accuracy_value
            accumulated_top1_accuracy_100_steps += top1_accuracy_value
            accumulated_top5_accuracy_10_steps += top5_accuracy_value
            accumulated_top5_accuracy_100_steps += top5_accuracy_value

            # The first step is slower since we have to wait until the examples queue has over min_examples
            # so we will not log the throughput at step 0
            if step == 0:
                continue

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = '%s: step %d, loss = %.2f, top-1 = %.3f%%, top-5 = %.3f%% ' \
                             '(%.1f examples/sec; %.3f sec/batch)'
                tf.logging.info(format_str % (datetime.datetime.now(), g_step, loss_value,
                                              accumulated_top1_accuracy_10_steps * 10,
                                              accumulated_top5_accuracy_10_steps * 10,
                                              examples_per_sec, sec_per_batch))
                accumulated_top1_accuracy_10_steps = 0.
                accumulated_top5_accuracy_10_steps = 0.

            if step % 100 == 0:

                save_accuracy(g_step, accumulated_top1_accuracy_100_steps,
                                    accumulated_top5_accuracy_100_steps);

                # Build the summary operation from the last tower summaries.
                summary_op = tf_.merge_summary(summaries)
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, g_step - 1)

                accumulated_top1_accuracy_100_steps = 0.
                accumulated_top5_accuracy_100_steps = 0.

            # Save the model checkpoint periodically.
            maybe_save_model(sess, saver, step, global_step)

            # Evaluate the model periodically
            maybe_submit_evaluation_job(step)


def main(argv=None):
    init_train_dir()
    train()


if __name__ == '__main__':
    tf.app.run()
