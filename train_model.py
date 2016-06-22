#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import pickle
import time
import datetime
from model.cnn_model import TextCNN
from tensorflow.core.framework import summary_pb2


# Credits to
# http://stackoverflow.com/questions/35714995/computing-exact-moving-average-over-multiple-batches-in-tensorflow
def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def batch_iter(data, batch_size, num_epochs):
    """ Generates a batch iterator for a data set."""
    data_size = len(data)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs): # Shuffle the data at each epoch
        shuffled_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffled_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

# ==============================================================================
# Parameters
# ==============================================================================
# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 10)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("output_every", 1, "Output current training error after this many steps (default: 1)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Adam Optimizer learning rate (default: 1e-4)")
tf.flags.DEFINE_float("dev_ratio", 0.1, "Percentage of data used for validation. Between 0 and 1. (default: 0.1)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# File paths
tf.flags.DEFINE_string("data_root", "./data", "Location of data folder.")
tf.flags.DEFINE_string("data_prefix", "full", "Prefix of preprocessed files (e.g. trainX, trainY, embeddings, etc.)."
                                              " Useful for differentiating between inputs containing *all* training"
                                              " data, and ones computed only over a subset.")
FLAGS = tf.flags.FLAGS

def dump_all_flags():
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

dump_all_flags()

# ==============================================================================
# Data Preparation
# ==============================================================================
pp = os.path.join(FLAGS.data_root, 'preprocessing')
prefix = FLAGS.data_prefix

print("Data root is: {0}.".format(FLAGS.data_root))
print("Actual files in: {0}.".format(pp))
print("Using input data file prefix: {0}.".format(prefix))

with open(os.path.join(pp, '{0}-vocab.pkl'.format(prefix)), 'rb') as f:
    vocabulary = pickle.load(f)
with open(os.path.join(pp, '{0}-vocab-inv.pkl'.format(prefix)), 'rb') as f:
    vocabulary_inv = pickle.load(f)

x = np.load(os.path.join(pp, '{0}-trainX.npy'.format(prefix)))
y = np.load(os.path.join(pp, '{0}-trainY.npy'.format(prefix)))
embeddings = np.load(os.path.join(pp, '{0}-embeddings.npy'.format(prefix)))

# Randomly shuffle data
np.random.seed(datetime.datetime.now().microsecond)
n_data = len(y)
shuffle_indices = np.random.permutation(np.arange(n_data))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# TODO: This is very crude, should use cross-validation

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
# TODO(andrei): On Euler we normally have ~40Gb of RAM. Could we increase this
# number then?
test_split = 200
x_dev = np.split(x_dev, test_split)
y_dev = np.split(y_dev, test_split)

# Training
# ==================================================


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    embedding_dim = embeddings.shape[1]
    with sess.as_default():
        cnn = TextCNN(
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
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        # out_dir = os.path.abspath(os.path.join(os.path.curdir, "./data/runs", timestamp))
        out_dir = os.path.abspath(os.path.join(FLAGS.data_root, 'runs', timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables and override pre-computed embeddings
        sess.run(tf.initialize_all_variables())
        sess.run(cnn.embeddings.assign(embeddings))
        # sess.run(cnn.embeddings_static.assign(embeddings))

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, train_loss, train_accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            train_summary_writer.add_summary(summaries, step)
            return train_loss, train_accuracy

        def dev_step(x_batch, y_batch):
            """Performs a model evaluation batch step on the dev set."""
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            _, dev_loss, dev_accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy], feed_dict)
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
            print(
                "\tloss {:g}\n\tacc {:g} (stddev {:g})\n\t(Tested on the full test set)\n"
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
                    evaluate_model(current_step)

                if current_step % FLAGS.checkpoint_every == 0:
                    print("Save model parameters...")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
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
                print("You interrupted the training. Performing final checkpoint.")
                print("Press C-c again to forcefully interrupt this.")
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
