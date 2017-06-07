"""
Evaluates the fusion model generated during training.

This code is based on the Inception tutorial in the tensorflow/models repository.

Usage example:

        python3 03_evaluate_softmax_model_fusion.py \
        --batch_size 23 \
        --image_size  $IMAGE_SIZE \
        --run_once $RUN_ONCE \
        --eval_data $EVAL_DATA \
        --data_dir $DATA_DIR \
        --cnn $CNN \
        --eval_dir $EVAL_DIR \
        --checkpoint_dir $CHECKPOINT_DIR \
        --light_summary_dir $LIGHT_SUMMARY_DIR \
        --weight_decay_rate $WEIGHT_DECAY_RATE \
        --label 0 \
        --net_type $NET_TYPE \
        --logits_output_file $LOGITS_OUTPUT_FILE

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

import pickle


tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('net_type', None, "Visual or semantic net.")
tf.app.flags.DEFINE_float('weight_decay_rate', 0.0001, """Weight decay rate.""")

def eval_once(saver, summary_writer, top_1_op, top_5_op, top_10_op, summary_op, num_examples,
              logits_op, labels_op, filenames_op, contr_fusion_op):
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

            while step < num_iter and not coord.should_stop():
                predictions, predictions_top5, predictions_top10, logits, labels, filenames, contributions = \
                    sess.run([top_1_op, top_5_op, top_10_op, logits_op, labels_op, filenames_op, contr_fusion_op])

                for i in range(logits.shape[0]):
                    results_dict[filenames[i]] = (logits[i, :], labels[i], contributions[i,:,:])

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

        # Compute precision @ 1, 5, 10.
        precision = true_count / total_sample_count
        precision_top5 = true_count_top5 / total_sample_count
        precision_top10 = true_count_top10 / total_sample_count
        tf.logging.info('%s: top-1 accuracy: %.3f\ttop-5 accuracy:%.3f\ttop-10 accuracy:%.3f'
                        % (datetime.now(), precision, precision_top5, precision_top10))

        write_accuracy_summaries(sess, summary_writer, summary_op, global_step, precision, precision_top5,
                                 precision_top10)

        maybe_store_logits(results_dict)
        maybe_store_light_summary(global_step, precision, precision_top5, precision_top10)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Evaluate MVSO for a number of steps."""
    with tf.Graph().as_default() as g:

        batch_norm_params = {
            'is_training': True,
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        }

        # Get images and labels
        dataset = MTVSOData(subset=FLAGS.eval_data)
        images, labels, filenames = batch_generator_mvso.generate_batch(dataset, FLAGS.batch_size, train=False,
                                                                        image_processing_fn=preprocess_image,
                                                                        include_filename=True)

        # Build inference Graph.
        if FLAGS.cnn == 'resnet50':

            num_clases_noun = dataset.num_classes()[1]
            num_clases_adjective = dataset.num_classes()[2]

            logits_nouns, _ = resnet_v1_50(images, num_clases_noun,
                                           scope='resnet_nouns_v1_50',
                                           reuse=None,
                                           is_training=False,
                                           batch_norm_decay=0.997,
                                           batch_norm_epsilon=1e-5,
                                           batch_norm_scale=True)

            logits_adjectives, _ = resnet_v1_50(images, num_clases_adjective,
                                                scope='resnet_adjectives_v1_50',
                                                reuse=None,
                                                is_training=True,
                                                batch_norm_decay=0.997,
                                                batch_norm_epsilon=1e-5,
                                                batch_norm_scale=True)
        else:
            raise ValueError('The specified CNN architecture is not supported')


        # Deduct and devide by mean and variance calculated from the softmax layer at the output of the resnets
        # This number was calculated for both the test and the train set

        logits_nouns = (tf.nn.softmax(logits_nouns) - 0.0072675)/tf.sqrt(0.0020015)
        logits_adjectives = (tf.nn.softmax(logits_adjectives) - 0.0072675)/tf.sqrt(0.0020015)

        with tf.variable_scope("fusion"):
            fusion_fc = tf.contrib.layers.fully_connected(inputs=tf.concat([logits_nouns, logits_adjectives], axis=1),
                                                          num_outputs=1024,
                                                          weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                              FLAGS.weight_decay_rate),
                                                          activation_fn=tf.nn.relu,
                                                          scope="fc")

        with tf.variable_scope("linear_anp"):
            logits_anp = tf.contrib.layers.linear(inputs=fusion_fc,
                                                  num_outputs=dataset.num_classes()[0],
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                      FLAGS.weight_decay_rate),
                                                  scope="logits"
                                                  )


        _ , pred = tf.nn.top_k(logits_anp, k=5)
        pred = tf.one_hot(pred, dataset.num_classes()[0], axis=1)

        # Calculate deep taylor decomposition for the top 5 predictions and concatenate
        # them in order to save them in a dictionary later.

        ''' 
        IMPORTANT: The function that calculates deep taylor decomposition do not allow multiple dimensions in axis 
        2, because of this, each of the 5 dimensions will have to be calculated separately. 
        '''

        dtd_out0 = tf.nn.softmax(logits_anp) * pred[:,:,0]
        dtd_out1 = tf.nn.softmax(logits_anp) * pred[:,:,1]
        dtd_out2 = tf.nn.softmax(logits_anp) * pred[:,:,2]
        dtd_out3 = tf.nn.softmax(logits_anp) * pred[:,:,3]
        dtd_out4 = tf.nn.softmax(logits_anp) * pred[:,:,4]

        n_input = dataset.num_classes()[1] + dataset.num_classes()[2]

        c1 = tf.reshape(compute_backprop(dtd_out0, fusion_fc, logits_nouns, logits_adjectives), [FLAGS.batch_size,
                                                                                                 n_input, 1])
        c2 = tf.reshape(compute_backprop(dtd_out1, fusion_fc, logits_nouns, logits_adjectives), [FLAGS.batch_size,
                                                                                                 n_input, 1])
        c3 = tf.reshape(compute_backprop(dtd_out2, fusion_fc, logits_nouns, logits_adjectives), [FLAGS.batch_size,
                                                                                                 n_input, 1])
        c4 = tf.reshape(compute_backprop(dtd_out3, fusion_fc, logits_nouns, logits_adjectives), [FLAGS.batch_size,
                                                                                                 n_input, 1])
        c5 = tf.reshape(compute_backprop(dtd_out4, fusion_fc, logits_nouns, logits_adjectives), [FLAGS.batch_size,
                                                                                                 n_input, 1])
        # Concatenate the contributions for the top 5 predictions that will be saved in a dictionary
        contr_fusion = tf.concat([c1, c2, c3, c4, c5], axis=2)

        # Calculate predictions.
        top_1_op = top_k_accuracy(logits_anp, labels, 1)
        top_5_op = top_k_accuracy(logits_anp, labels, 5)
        top_10_op = top_k_accuracy(logits_anp, labels, 10)

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
                      logits_anp, labels, filenames, contr_fusion)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    init_eval_dir()
    evaluate()


if __name__ == '__main__':
    tf.app.run()
