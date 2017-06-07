import pickle
import os
import tensorflow as tf

from .tf_functions import *


tf.logging.set_verbosity(tf.logging.INFO)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', None,
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', None,
                           """Either 'train' or 'val'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', None,
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 0,
                            """Number of examples to run. Set to 0 to evaluate the whole set.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, """Moving average decay.""")
tf.app.flags.DEFINE_string('cnn', 'resnet50', """CNN architecture""")
tf.app.flags.DEFINE_boolean('remove_dir', False, """Whether to remove eval_dir before starting evaluation.""")

# Store outputs
tf.app.flags.DEFINE_string('logits_output_file', None,
                           """File where a pickled list with (logits, ground_truth) tuples for each image will be
                           stored""")

tf.app.flags.DEFINE_string('light_summary_dir', None, "Where the light summary is stored.")


def init_eval_dir():
    if tf.gfile.Exists(FLAGS.eval_dir) and FLAGS.remove_dir:
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    if not tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.MakeDirs(FLAGS.eval_dir)


def restore_model(sess, saver):
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        tf.logging.info('Restoring from %s' % ckpt.model_checkpoint_path)
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        tf.logging.info('No checkpoint file found')
        global_step = -1

    return global_step


def top_k_accuracy(logits, labels, k):
    _, topk_predictions_op = tf.nn.top_k(logits, k=k)
    topk_correct_pred = tf.cast(tf.equal(tf.transpose(topk_predictions_op), labels), tf.float32)
    return tf.reduce_sum(topk_correct_pred, 0)


def predicted_class(logits):
    return tf.argmax(logits, 1)


def write_accuracy_summaries(sess, summary_writer, summary_op, global_step, precision, precision_top5, precision_top10):
    summary = tf.Summary()
    summary.ParseFromString(sess.run(summary_op))
    summary.value.add(tag='accuracy_top-1', simple_value=100. * precision)
    summary.value.add(tag='accuracy_top-5', simple_value=100. * precision_top5)
    summary.value.add(tag='accuracy_top-10', simple_value=100. * precision_top10)
    summary_writer.add_summary(summary, global_step)


def maybe_store_logits(results_dict):
    if FLAGS.logits_output_file is not None:
        with open(FLAGS.logits_output_file, 'wb') as f:
            pickle.dump(results_dict, f, protocol=0)

def get_variables(scope_list):
    def _keep_var(v):
        for sc in scope_list:
            if sc in v.name:
                return True
        return False
    return [var for var in tf.trainable_variables() if _keep_var(var)]


def maybe_store_light_summary(global_step, precision, precision_top5, precision_top10):
    if FLAGS.light_summary_dir != None:
        summary_file = os.path.join(FLAGS.light_summary_dir, 'light_summary_test.txt')
        with open(summary_file, 'a') as f:
            string = str(global_step) + " " + str(int(precision * 1000)) + " " + str(int(precision_top5 * 1000)) \
                     + " " + str(int(precision_top10 * 1000))
            f.write(string + "\n")


# Compute propagation rule for layers with inputs in the positive domain
def z_plus_rule(layer_output, layer_input, layer_vars, epsilon=1e-09):
    layer_weights = extract_var(layer_vars, "weights")
    layer_biases = extract_var(layer_vars, "biases")

    v = tf.maximum(tf.zeros(layer_weights.get_shape()), layer_weights)
    b = tf.maximum(tf.zeros(layer_biases.get_shape()), layer_biases)
    z = tf.matmul(layer_input, v) + b + epsilon
    s = layer_output / z
    c = tf.matmul(s, tf.transpose(v))
    f = layer_input * c

    return f


# Compute propagation rule for layers with bounded inputs
def z_b_rule(layer_output, layer_input, layer_vars, lower_bound, upper_bound, epsilon=1e-09):
    layer_weights = extract_var(layer_vars, "weights")
    layer_biases = extract_var(layer_vars, "biases")

    b = tf.maximum(tf.zeros(layer_biases.get_shape()), layer_biases)

    w, v = layer_weights, tf.maximum(tf.zeros(layer_weights.get_shape()), layer_weights)
    u = tf.minimum(tf.zeros(layer_weights.get_shape()), layer_weights)
    x, l, h = layer_input, layer_input * 0 + lower_bound, layer_input * 0 + upper_bound

    z = tf.matmul(x, w) - tf.matmul(l, v) - tf.matmul(h, u) + epsilon
    s = layer_output / (z + b)
    f = x * tf.matmul(s, tf.transpose(w)) - l * tf.matmul(s, tf.transpose(v)) - h * tf.matmul(s, tf.transpose(u))
    return f


# Compute complete backprop from output logits to nouns and adjectives outputs
def compute_backprop(dtd_out, fusion_fc, logits_nouns, logits_adjectives):
    contr_linear_anp = z_plus_rule(layer_output=dtd_out,
                                   layer_input=fusion_fc,
                                   layer_vars=get_variables(["linear_anp"]))

    contr_fusion = z_b_rule(layer_output=contr_linear_anp,
                            layer_input=tf.concat([logits_nouns, logits_adjectives], axis=1),
                            layer_vars=get_variables(["fusion/fc"]),
                            lower_bound=(- 0.0072675) / tf.sqrt(0.0020015),
                            upper_bound=1)

    return contr_fusion


def extract_var(layer_weights, name):
    for sc in layer_weights:
        if name in sc.name:
            layer_weights = sc
            break

    assert name in layer_weights.name
    return layer_weights


def create_saver():
    """ Restore the moving average version of the learned variables for eval. """
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    return tf.train.Saver(variables_to_restore)


