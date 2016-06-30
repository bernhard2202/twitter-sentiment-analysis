#! /usr/bin/env python

import os
import pickle
import time
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

from model.cnn_model import TextCNN
from model.lstm import TextLSTM
from model.util import batch_iter

# Credits to
# http://stackoverflow.com/questions/35714995/computing-exact-moving-average-over-multiple-batches-in-tensorflow
def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


# ==============================================================================
# Parameters
# ==============================================================================
# Model Hyperparameters
# General
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability."
                                                " Lower means stronger"
                                                " regularization."
                                                " (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# CNN-specific
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")

# LSTM-specific
tf.flags.DEFINE_integer("lstm_hidden_size", 128, "Size of the hidden LSTM"
                                                 " layer.")
tf.flags.DEFINE_integer("lstm_hidden_layers", 1, "Number of hidden LSTM"
                                                 " layers.")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 10)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("output_every", 1, "Output current training error after this many steps (default: 1)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Adam Optimizer learning rate (default: 1e-4)")
tf.flags.DEFINE_float("dev_ratio", 0.1, "Percentage of data used for validation. Between 0 and 1. (default: 0.1)")
tf.flags.DEFINE_integer("test_split", 50,
                        "Number of splits of the test data to lower memory pressure during. (default:50)")
tf.flags.DEFINE_boolean("clip_gradients", False, "Whether to enable gradient"
                                                 " clipping (useful when"
                                                 " training complex LSTMs).")
tf.flags.DEFINE_float("clip_gradient_value", 5.0, "Gradient value to clip to"
                                                  " when 'clip_gradients' is"
                                                  " enabled.")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("label", "", "Additional label to append to experiment"
                                    " directory name such as whether the"
                                    " embeddings used were pre-trained, or"
                                    " GloVe vs. word2vec, etc.")

# File paths
tf.flags.DEFINE_string("data_root", "./data", "Location of data folder.")
tf.flags.DEFINE_string("data_prefix", "full",
                       "Prefix of preprocessed files (e.g. trainX, trainY, embeddings, etc.)."
                       " Useful for differentiating between inputs containing *all* training"
                       " data, and ones computed only over a subset.")
tf.flags.DEFINE_boolean("lstm", False,
                        "Whether to use the LSTM model. If set to False, uses"
                        " the CNN. (default: False)")
FLAGS = tf.flags.FLAGS


def dump_all_flags():
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")


def flags_to_string():
    FLAGS._parse_flags()
    res = ""
    for attr, value in sorted(FLAGS.__flags.items()):
        res += "{}={}\n".format(attr.upper(), value)
    return res


dump_all_flags()

# ==============================================================================
# Data Preparation
# ==============================================================================
pp = os.path.join(FLAGS.data_root, 'preprocessing')
prefix = FLAGS.data_prefix

print("Data root is: {0}.".format(FLAGS.data_root))
print("Actual preprocessed input files in: {0}.".format(pp))
print("Using input data file prefix: {0}.".format(prefix))

print("Loading data...")
with open(os.path.join(pp, '{0}-vocab.pkl'.format(prefix)), 'rb') as f:
    vocabulary = pickle.load(f)

x = np.load(os.path.join(pp, '{0}-trainX.npy'.format(prefix)))
y = np.load(os.path.join(pp, '{0}-trainY.npy'.format(prefix)))
embeddings = np.load(os.path.join(pp, '{0}-embeddings.npy'.format(prefix)))
print("Finished loading input X, Y, embeddings, and vocabulary.")

# Randomly shuffle data
random_seed = datetime.datetime.now().microsecond
print("Using random seed: {0}".format(random_seed))
np.random.seed(random_seed)
n_data = len(y)
shuffle_indices = np.random.permutation(np.arange(n_data))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# TODO: This is very crude, should use cross-validation

# Train/dev split
n_dev = int(FLAGS.dev_ratio * n_data)
n_train = n_data - n_dev
x_train, x_dev = x_shuffled[:n_train], x_shuffled[n_train:]
y_train, y_dev = y_shuffled[:n_train], y_shuffled[n_train:]

assert len(x_train) + len(x_dev) == n_data
assert len(y_train) + len(y_dev) == n_data

print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# This splits the test data into chunks to lower memory pressure during
# validation.
test_split = FLAGS.test_split
x_dev = np.array_split(x_dev, test_split)
y_dev = np.array_split(y_dev, test_split)

# ==================================================
# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    embedding_dim = embeddings.shape[1]
    with sess.as_default():
        # TODO(andrei): Warn on all unused flags (e.g. set CNN options when
        # using an LSTM).
        if FLAGS.lstm:
            print("\nUsing LSTM.")
            model = TextLSTM(sequence_length=x_train.shape[1],
                             vocab_size=len(vocabulary),
                             embedding_size=embedding_dim,
                             hidden_size=FLAGS.lstm_hidden_size,
                             layer_count=FLAGS.lstm_hidden_layers)
        else:
            print("\nUsing CNN.")
            model = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=2,
                vocab_size=len(vocabulary),
                embedding_size=embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        # 'compute_gradients' returns a list of (gradient, variable) pairs.
        grads_and_vars = optimizer.compute_gradients(model.loss)

        # If enabled, we apply gradient clipping, in order to deal with the
        # exploding gradient problem common when training LSTMs.
        # TODO(andrei): Gradient magnitude tracing to see effectiveness of
        # clipping.
        if FLAGS.clip_gradients:
            print("Will clip gradients |.| < {0}"
                  .format(FLAGS.clip_gradient_value))

            if not FLAGS.lstm:
                raise ValueError("Gradient clipping should probably not be"
                                 " used with CNNs.")

            # Note that this is much less fancy than it looks. We don't do L2
            # regularization, we don't compute the L2 norm of the gradient; we
            # simply truncate its raw value.
            def tf_clip(gradient):
                if gradient is None:
                    # Workaround for a particular case where a variable's
                    # gradient was returned 'None' by 'compute_gradients'.
                    # TODO(andrei): Investigate whether this matters.
                    return None
                return tf.clip_by_value(gradient,
                                        -FLAGS.clip_gradient_value,
                                        FLAGS.clip_gradient_value)
            grads_and_vars = [(tf_clip(grad), var)
                              for grad, var in grads_and_vars]
        else:
            print("Will NOT perform gradient clipping.")

        train_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=global_step)

        # Keep track of gradient values and sparsity (time intense!)
        # grad_summaries = []
        # for g, v in grads_and_vars:
        #     if g is not None:
        #         grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
        #         sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        #         grad_summaries.append(grad_hist_summary)
        #         grad_summaries.append(sparsity_summary)
        # grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = '{0}-w2v-{1}d-{2}-{3}'.format(
            'lstm' if FLAGS.lstm else 'cnn',
            embedding_dim,
            timestamp,
            FLAGS.label)
        out_dir_full = os.path.abspath(os.path.join(FLAGS.data_root, 'runs', out_dir))
        os.mkdir(out_dir_full)
        meta_fname = os.path.join(out_dir_full, 'meta.txt')
        print("Writing to {}\n".format(out_dir_full))
        print("Meta-information will be written to {}.".format(meta_fname))

        with open(meta_fname, 'w') as meta_file:
            # TODO(andrei): Add whatever additional information is necessary.
            meta_file.write("Meta-information\n")
            meta_file.write("Label: {0}\n".format(FLAGS.label))
            if FLAGS.lstm:
                meta_file.write("LSTM\n")
            else:
                meta_file.write("CNN\n")
            meta_file.write("\nFlags:\n")
            meta_file.write(flags_to_string())

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", model.loss)
        acc_summary = tf.scalar_summary("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir_full, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_dir = os.path.join(out_dir_full, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir_full, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables and override pre-computed embeddings
        sess.run(tf.initialize_all_variables())
        sess.run(model.embeddings.assign(embeddings))
        # sess.run(cnn.embeddings_static.assign(embeddings))

        def train_step(x_batch, y_batch):
            """A single training step"""
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch,
              model.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            # add grad_summaries_merged to keep track of gradient values (time intense!)
            _, step, summaries, train_loss, train_accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            train_summary_writer.add_summary(summaries, step)
            return train_loss, train_accuracy

        def dev_step(x_batch, y_batch):
            """Performs a model evaluation batch step on the dev set."""
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch,
              model.dropout_keep_prob: 1.0
            }
            _, dev_loss, dev_accuracy = sess.run(
                [global_step, model.loss, model.accuracy], feed_dict)
            return dev_loss, dev_accuracy

        def evaluate_model(current_step):
            """Evaluates model on a dev set."""
            losses = np.empty(test_split)
            accuracies = np.empty(test_split)

            for i in range(test_split):
                loss, accuracy = dev_step(x_dev[i], y_dev[i])
                losses[i] = loss
                accuracies[i] = accuracy

            average_loss = np.nanmean(losses)
            average_accuracy = np.nanmean(accuracies)
            std_accuracy = np.nanstd(accuracies)
            dev_summary_writer.add_summary(make_summary('accuracy', average_accuracy),
                                           current_step)
            dev_summary_writer.add_summary(make_summary('loss', average_loss),
                                           current_step)
            dev_summary_writer.add_summary(make_summary('accuracy_std', std_accuracy),
                                           current_step)
            time_str = datetime.datetime.now().isoformat()
            print("{}: Evaluation report at step {}:".format(time_str, current_step))
            print("\tloss {:g}\n\tacc {:g} (stddev {:g})\n"
                  "\t(Tested on the full test set)\n"
                  .format(average_loss, average_accuracy, std_accuracy))

        # Generate batches
        batches = batch_iter(
            np.array(list(zip(x_train, y_train))),
            FLAGS.batch_size,
            FLAGS.num_epochs)
        # Training loop. For each batch...
        current_step = None
        try:
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                l, a = train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.output_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, l, a))

                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluating...")
                    eval_start_ms = int(time.time() * 1000)
                    evaluate_model(current_step)
                    eval_time_ms = int(time.time() * 1000) - eval_start_ms
                    print("Evaluation performed in {0}ms.".format(eval_time_ms))

                if current_step % FLAGS.checkpoint_every == 0:
                    print("Save model parameters...")
                    path = saver.save(sess, checkpoint_prefix,
                                      global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            if current_step is None:
                print("No steps performed.")
            else:
                print("\n\nFinished all batches. Performing final evaluations.")

                print("Performing final evaluation...")
                evaluate_model(current_step)

                print("Performing final checkpoint...")
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

                print("Here's all the flags again:")
                dump_all_flags()

        except KeyboardInterrupt:
            if current_step is None:
                print("No checkpointing to do.")
            else:
                # TODO(andrei): Consider also evaluating here.
                print("Training interrupted. Performing final checkpoint.")
                print("Press C-c again to forcefully interrupt this.")
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
