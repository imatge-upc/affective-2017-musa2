"""
Evaluates the softmax baseline models generated during training.

This code is based on the Inception tutorial in the tensorflow/models repository.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import util.tf_functions as tf_

from util.evaluation import *
from util.nn import resnet_v1_50
from util.mtvso_data import MTVSOData
from util import batch_generator_mvso
from util.vgg_preprocessing import preprocess_image


tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS


def eval_once(saver, summary_writer, top_1_op, top_5_op, top_10_op, summary_op, num_examples,
              logits_op, labels_op, filenames_op, mean, var):
    """
    Run Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_1_op: Top 1 op.
      top_5_op: Top 5 op.
      top_10_op: Top 10 op.
      summary_op: Summary op.
      num_examples: number of samples in the evaluation set
      logits_op: output of the model
      labels_op: ground truth
      filenames_op: filename for each example
    """
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        global_step = restore_model(sess, saver)

        if int(global_step) < 0:
            return

        # Store the outputs
        results_dict = {}

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(num_examples / FLAGS.batch_size))
            true_count, true_count_top5, true_count_top10 = 0, 0, 0  # Count the number of top-k correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0

            total_mean = 0
            total_var = 0
            while step < num_iter and not coord.should_stop():
                predictions, predictions_top5, predictions_top10, logits, labels, filenames, m, v = \
                    sess.run([top_1_op, top_5_op, top_10_op, logits_op, labels_op, filenames_op, mean, var])
                for i in range(logits.shape[0]):
                    results_dict[filenames[i]] = (logits[i, :], labels[i])

                total_mean +=m
                total_var += v
                true_count += np.sum(predictions)
                true_count_top5 += np.sum(predictions_top5)
                true_count_top10 += np.sum(predictions_top10)
                step += 1
                tf.logging.info(('Step: %d/%d  --  top-1 accuracy: %.3f%%\ttop-5 accuracy:%.3f%%'
                                 '\ttop-10 accuracy: %.3f%%')
                                % (step, num_iter,
                                   100. * true_count / (step * FLAGS.batch_size),
                                   100. * true_count_top5 / (step * FLAGS.batch_size),
                                   100. * true_count_top10 / (step * FLAGS.batch_size)))
        except Exception as e:
            coord.request_stop(e)

        total_mean /= num_iter
        total_var /= num_iter
        # Compute precision @ 1, 5, 10.
        precision = true_count / total_sample_count
        precision_top5 = true_count_top5 / total_sample_count
        precision_top10 = true_count_top10 / total_sample_count
        tf.logging.info('%s: top-1 accuracy: %.3f\ttop-5 accuracy:%.3f\ttop-10 accuracy:%.3f'
                        % (datetime.now(), precision, precision_top5, precision_top10))

        tf.logging.info('Total mean = %f, Total variance =%f' % (total_mean, total_var))

        write_accuracy_summaries(sess, summary_writer, summary_op, global_step, precision, precision_top5,
                                 precision_top10)

        maybe_store_logits(results_dict)
        maybe_store_light_summary(global_step, precision, precision_top5, precision_top10)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Evaluate MVSO for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels
        dataset = MTVSOData(subset=FLAGS.eval_data)
        images, labels, filenames = batch_generator_mvso.generate_batch(dataset, FLAGS.batch_size, train=False,
                                                                        image_processing_fn=preprocess_image,
                                                                        include_filename=True)

        # Build a Graph that computes the logits predictions from the inference model.
        if FLAGS.cnn == 'resnet50':
            logits, _ = resnet_v1_50(images, dataset.num_classes()[FLAGS.label],
                                     scope='resnet_v1_50',  #
                                     reuse=None,
                                     is_training=False,
                                     batch_norm_decay=0.997,
                                     batch_norm_epsilon=1e-5,
                                     batch_norm_scale=True)
        else:
            raise ValueError('The specified CNN architecture is not supported')

        softmax = tf.nn.softmax(logits)
        tf.logging.info("Shape of logits is:" + str(logits.get_shape()))
        mean, var = tf.nn.moments(softmax, axes=[0,1])

        # Calculate predictions.
        top_1_op = top_k_accuracy(logits, labels, 1)
        top_5_op = top_k_accuracy(logits, labels, 5)
        top_10_op = top_k_accuracy(logits, labels, 10)

        # Restore the moving average version of the learned variables for eval.
        saver = create_saver()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf_.merge_all_summaries()
        summary_writer = tf_.summary_writer(FLAGS.eval_dir, g)

        while True:
            if FLAGS.num_examples == 0:
                num_examples = dataset.num_examples_per_epoch()
            else:
                num_examples = FLAGS.num_examples
            eval_once(saver, summary_writer, top_1_op, top_5_op, top_10_op, summary_op, num_examples,
                      logits, labels, filenames, mean, var)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    init_eval_dir()
    evaluate()


if __name__ == '__main__':
    tf.app.run()
