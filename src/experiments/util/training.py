"""
Set of functions and flags used for training neural networks

Some of them are taken from TensorFlow multi-GPU tutorial, e.g. average_gradients()
"""

import os
import subprocess
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .tf_functions import *

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', None,
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1, """Initial learning rate value.""")
tf.app.flags.DEFINE_boolean('exponential_decay', False, """Whether to use exponential LR decay.""")
tf.app.flags.DEFINE_integer('lr_epochs_per_decay', 10, """Number of epochs between lr decays""")
tf.app.flags.DEFINE_float('lr_decay_factor', 0.9, """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, """Moving average decay.""")
tf.app.flags.DEFINE_float('weight_decay_rate', 0.0001, """Weight decay rate.""")
tf.app.flags.DEFINE_string('cnn', 'resnet50', """CNN architecture""")
tf.app.flags.DEFINE_boolean('remove_dir', False,
                            """Whether to remove train_dir before starting training.""")

tf.app.flags.DEFINE_string('checkpoint', None, """Checkpoint file with the pre-trained model weights""")
tf.app.flags.DEFINE_string('checkpoint_adj', None, """Checkpoint file with the pre-trained model weights""")
tf.app.flags.DEFINE_string('checkpoint_noun', None, """Checkpoint file with the pre-trained model weights""")

tf.app.flags.DEFINE_boolean('restore_logits', False, """Whether to restore logits when loading a pre-trained model.""")
tf.app.flags.DEFINE_string('optimizer', 'RMSProp', """SGD, Momentum, RMSProp, Adam""")
tf.app.flags.DEFINE_boolean('resume_training', False, """Resume training from last checkpoint in train_dir""")
tf.app.flags.DEFINE_boolean('histograms', False, """Whether to store variable histograms summaries.""")
tf.app.flags.DEFINE_integer('eval_interval_iters', 3000, """How often to run the eval, in training steps.""")

# Job file to submit after finishing
tf.app.flags.DEFINE_string('evaluation_job', None, """Path to the cmd file that performs the evaluation""")

tf.app.flags.DEFINE_string('light_summary_dir', None, "Where the light summary is stored.")



def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = concat(grads, 0, 'concat_logits')
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def cross_entropy_loss(logits, labels):
    """
    Add L2Loss to all the trainable variables.
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def create_optimizer(learning_rate):
    if FLAGS.optimizer.lower() == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    elif FLAGS.optimizer.lower() == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    elif FLAGS.optimizer.lower() == 'rmsprop':
        # Parameters from 'Rethinking Inception Architecture for Computer Vision'
        opt = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9, epsilon=1.0)
    elif FLAGS.optimizer.lower() == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate)
    else:
        raise AttributeError('The specified optimizer is not supported')
    return opt


def create_learning_rate_scheduler(global_step, dataset=None):
    if FLAGS.exponential_decay:  # Decay the learning rate exponentially based on the number of steps.
        assert dataset is not None, 'Exponential decay depends on the dataset'
        num_batches_per_epoch = (dataset.num_examples_per_epoch() / FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * FLAGS.lr_epochs_per_decay)
        tf.logging.info('Using exponential learning rate decay')
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.lr_decay_factor,
                                        staircase=True)
    else:  # Constant learning rate
        tf.logging.info('Using constant learning rate of %f', FLAGS.initial_learning_rate)
        lr = tf.get_variable(
            'learning_rate', [],
            initializer=tf.constant_initializer(FLAGS.initial_learning_rate), trainable=False)

    return lr


def overwrite_learning_rate(sess, lr):
    if not FLAGS.exponential_decay and FLAGS.resume_training:
        tf.logging.info('Overwriting stored learning rate. Using constant learning rate of %f',
                        FLAGS.initial_learning_rate)
        sess.run(lr.assign(FLAGS.initial_learning_rate))


def top_k_accuracy(logits_op, labels_op, k):
    if k == 1:
        top1_predictions_op = tf.argmax(logits_op, 1)
        top1_correct_pred = tf.equal(top1_predictions_op, tf.cast(labels_op, tf.int64))
        return tf.reduce_mean(tf.cast(top1_correct_pred, tf.float32))
    else:
        _, topk_predictions_op = tf.nn.top_k(logits_op, k=k)
        topk_correct_pred = tf.cast(tf.equal(tf.transpose(topk_predictions_op), tf.cast(labels_op, tf.int32)),
                                    tf.float32)
        return tf.reduce_mean(tf.reduce_sum(topk_correct_pred, 0), 0)


def maybe_submit_evaluation_job(step):

    if (step % FLAGS.eval_interval_iters == 0 or (step + 1) == FLAGS.max_steps) and step > 0:
        if FLAGS.evaluation_job is not None:
            tf.logging.info("Executing evaluation script...")
            # tf.logging.info(subprocess.check_output([FLAGS.evaluation_job], shell=True))


def init_train_dir():
    if tf.gfile.Exists(FLAGS.train_dir) and FLAGS.remove_dir:
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)


def maybe_track_vars_and_gradients(grads, summaries):
    if FLAGS.histograms:
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))


def maybe_save_model(sess, saver, step, global_step):
    if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        tf.logging.info('Saving model checkpoint')
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, int(sess.run(global_step)))


def save_accuracy(global_step, precision_top1, precision_top5):
    if FLAGS.light_summary_dir is not None:
        summary_file = os.path.join(FLAGS.light_summary_dir, 'light_summary_train.txt')
        with open(summary_file, 'a') as f:
            string = str(int(global_step)) + " " + str(int(precision_top1*10)) \
                     + " " + str(int(precision_top5*10))
            f.write(string + "\n")


def _get_variables_to_restore(scope):

    variables_to_restore = []
    for op in slim.get_model_variables(scope):
        if FLAGS.restore_logits or not op.name.__contains__('logits'): variables_to_restore.append(op)
    return variables_to_restore


def generate_variable_dict(current_scope, checkpoint_scope):
    variable_list = [v for v in tf.trainable_variables() if current_scope in v.name]
    return {var.op.name.replace(current_scope, checkpoint_scope): var for var in variable_list}


# Restore the parameters from older checkpoints if resumed training or from adjective and noun nets
def restore_model(sess, saver, current_scope=None, checkpoint_scope=None):
    directory = None

    if FLAGS.resume_training:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

    elif FLAGS.checkpoint is not None:
        assert current_scope is not None, 'scope is required to restore from a pre-trained model'

        if checkpoint_scope is None:
            variables_to_restore = _get_variables_to_restore(current_scope)
            directory = FLAGS.checkpoint
        else:
            if current_scope == "resnet_adjectives_v1_50":
                directory = FLAGS.checkpoint_adj
            if current_scope == "resnet_nouns_v1_50":
                directory = FLAGS.checkpoint_noun
            variables_to_restore = generate_variable_dict(current_scope, checkpoint_scope)

        tf.logging.info("Directory to read: " + directory)
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, directory)


def get_variables(scope_list):

    def _keep_var(v):
        for sc in scope_list:
            if sc in v.name:
                return True
        return False
    return [var for var in tf.trainable_variables() if _keep_var(var)]