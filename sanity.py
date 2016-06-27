"""Utility for double checking a model's train/dev accuracy.

TODO(andrei): Also support evaluating ensembles.
"""

import numpy as np
try:
    from sklearn.utils import shuffle
except ImportError:
    print("This tool requires 'scikit-learn'. If you're on Euler, it's "
          "probably not worth your time to install.")
    exit(0)
import tensorflow as tf

from typing import Tuple
import os
import random
import time


random_seed = 0x123
random.seed(random_seed)
np.random.seed(random_seed)

# Example: './data/runs/euler/local-w2v-275d-1466050948/checkpoints/model-96690'
tf.flags.DEFINE_string("checkpoint_file", None, "Checkpoint file from the"
                                                " training run.")
tf.flags.DEFINE_string("second_checkpoint_file", None, "Another checkpoint"
                                                       " file for computing"
                                                       " avg. probabilities.")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if FLAGS.checkpoint_file is None:
    raise ValueError("Please specify a TensorFlow checkpoint file to use for"
                     " making the predictions (--checkpoint_file <file>).")

if FLAGS.second_checkpoint_file is not None:
    print("Using poor man's ensemble.")
    second_cp = FLAGS.second_checkpoint_file
else:
    second_cp = None

# How many data points to evaluate.
evlim = 5000

print("Will shuffle data data and limit it to {0} points.".format(evlim))

if FLAGS.checkpoint_file is None:
    raise ValueError("Please specify a TensorFlow checkpoint file to use for"
                     " making the predictions (--checkpoint_file <file>).")

checkpoint_file = FLAGS.checkpoint_file
timestamp = int(time.time())
default_pp = os.path.join('data', 'preprocessing')


class ModelConfig(object):

    def __init__(self,
                 checkpoint_fname,
                 trainx_fname=os.path.join(default_pp, 'full-trainX.npy'),
                 trainy_fname=os.path.join(default_pp, 'full-trainY.npy'),
                 valid_fname=os.path.join(default_pp, 'validateX.npy'),
                 input_x_name='input_x',
                 input_y_name='input_y',
                 dropout_name='dropout_keep_prob',
                 predictions_name='output/predictions',
                 accuracies_name='accuracy/accuracy'):
        self.checkpoint_fname = checkpoint_fname
        self.trainx_fname = trainx_fname
        self.trainy_fname = trainy_fname
        self.valid_fname = valid_fname

        self.input_x_name = input_x_name
        self.input_y_name = input_y_name
        self.dropout_name = dropout_name
        self.predictions_name = predictions_name
        self.accuracies_name = accuracies_name

        self.trainx = np.load(trainx_fname)
        self.trainy = np.load(trainy_fname)

        # Shuffle the data in a predictable way, so that multiple configs
        # load different data files (preprocessed differently) but shuffle them
        # in an identical way.
        self.trainx, self.trainy = shuffle(self.trainx, self.trainy,
                                           random_state=random_seed)
        self.trainx = self.trainx[:evlim]
        self.trainy = self.trainy[:evlim]


def evaluate(config: ModelConfig) -> Tuple[float, np.ndarray]:
    """Loads and evaluates the model specified in the config.

    Returns
        A pair consisting of the accuracy of the classifier and a 2d numpy array
        consisting of each input's probabilities.
    """
    graph = tf.Graph()
    with graph.as_default(), tf.Session() as sess:
        print("Loading saved meta graph from checkpoint file {0}."
              .format(config.checkpoint_fname))
        saver = tf.train.import_meta_graph("{}.meta".format(config.checkpoint_fname))
        print("Restoring variables...")
        saver.restore(sess, config.checkpoint_fname)
        print("Finished TF graph load.")

        # Input placeholders
        input_x = graph.get_operation_by_name(config.input_x_name).outputs[0]
        input_y = graph.get_operation_by_name(config.input_y_name).outputs[0]

        if config.dropout_name is not None:
            dropout_keep_prob = graph.get_operation_by_name(config.dropout_name).outputs[0]
        else:
            dropout_keep_prob = None

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name(config.predictions_name).outputs[0]
        accuracy = graph.get_operation_by_name(config.accuracies_name).outputs[0]

        feed_dict = {input_x: config.trainx, input_y: config.trainy}
        if dropout_keep_prob is not None:
            feed_dict[dropout_keep_prob] = 1.0

        acc, predictions_out = sess.run([accuracy, predictions], feed_dict)
        return acc, predictions_out


def manual_evaluate(config: ModelConfig) -> float:
    """Evaluates the model's accuracy iteratively.

    Much slower than 'evaluate'.

    Returns:
        The accuracy of the model specified by the config.
    """
    # TODO(andrei): Nice error bar because you have nothing better to do you
    # worthless lazy procrastinating sack of shit.
    # TODO(andrei): Fix self-esteem issues.

    # TODO(andrei): Clean up this method if it's worth it.
    graph = tf.Graph()
    with graph.as_default(), tf.Session() as sess:
        print("Loading saved meta graph from checkpoint file {0}."
              .format(checkpoint_file))
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        print("Restoring variables...")
        saver.restore(sess, checkpoint_file)
        print("Finished TF graph load.")

        all_predictions = []
        print("Computing accuracy in the old-fashioned way...")
        print("Consider interrupting execution unless you're paranoid.")
        for (id, row) in enumerate(validation_data):
            if (id + 1) % 1000 == 0:
                print("Done tweets: {0}/{1}".format(id + 1, len(validation_data)))

            prediction = sess.run(predictions, {
                input_x: [row],
                dropout_keep_prob: 1.0
            })[0]
            all_predictions.append((id, prediction))

        print("Prediction done")
        correct = 0
        zero_preds = 0
        for id, pred in all_predictions:
            if id > evlim:
                break

            # print(id, pred)
            if pred[0] >= pred[1]:
                zero_preds += 1

            if (pred[0] >= pred[1] and validation_data_check[id][0] == 1) \
                    or (pred[0] < pred[1] and validation_data_check[id][1] == 1):
                correct += 1

        acc = correct * 1.0 / evlim
        print("We had {0} cases with pred[0] >= pred[1] out of {1}.".format(
            zero_preds, evlim))
        return acc


config_A = ModelConfig(
    FLAGS.checkpoint_file,
    trainx_fname='/tmp/cil/full-trainX.npy',
    trainy_fname='/tmp/cil/full-trainY.npy',
    input_x_name='input_batch_x',
    input_y_name='input_batch_x_1',
    predictions_name='output/Softmax',
    accuracies_name='accuracy/Mean')

config_B = ModelConfig(FLAGS.second_checkpoint_file)

acc_A, preds_A = evaluate(config_A)
print("Accuracy A using clean mode: {0}".format(acc_A))

acc_B, preds_B = evaluate(config_B)
print("Accuracy B using clean mode: {0}".format(acc_B))

exit(0)

# graph = tf.Graph()
# with graph.as_default():
#     sess = tf.Session()
#     with sess.as_default():
#         print("Loading saved meta graph...")
#         saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#         print("Restoring variables...")
#         saver.restore(sess, checkpoint_file)
#         print("Finished TF graph load.")
#
#         if second_cp is not None:
#             print("Loading second checkpoint.")
#
#         # aux = [o.name for o in graph.get_operations() if 'input' in o.name]
#         # print("Interesting ops:")
#         # print(aux)
#         # print()
#
#         # Get the placeholders from the graph by name
#         # If you forget to name your input, try 'Placeholder', or 'Placeholder_1'.
#         # input_x = graph.get_operation_by_name("Placeholder").outputs[0]
#         # input_x = graph.get_operation_by_name("input_x").outputs[0]
#         # input_y = graph.get_operation_by_name("input_y").outputs[0]
#
#         input_x = graph.get_operation_by_name("input_batch_x").outputs[0]
#         input_y = graph.get_operation_by_name("input_batch_x_1").outputs[0]
#
#         # input_y = graph.get_operation_by_name("Placeholder_1").outputs[0]
#         # input_y = graph.get_operation_by_name("input_batch_x_1").outputs[0]
#         dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
#
#         # Tensors we want to evaluate
#         predictions = graph.get_operation_by_name("output/Softmax").outputs[0]
#         # predictions = graph.get_operation_by_name("output/predictions").outputs[0]
#
#         accuracy = graph.get_operation_by_name('accuracy/Mean').outputs[0]
#         # accuracy = graph.get_operation_by_name('accuracy/accuracy').outputs[0]
#
#         acc, predictions_out = sess.run(
#             [accuracy, predictions],
#             feed_dict={
#                 input_x: validation_data,
#                 input_y: validation_data_check,
#                 dropout_keep_prob: 1.0,
#             })
#         print("Official accuracy:")
#         print(acc)
#
#         print("Official predictions we made:")
#         print(predictions_out)
#         # print("Exiting early.")
#         # exit(0)
#
#         try:
#             manual_evaluate(config)
#         except KeyboardInterrupt:
#             print("\nK, nvm.")
#
#         print("Finished evaluation of model from checkpoint file {0}.".format(
#             checkpoint_file))

# if second_cp is not None:
#     g2 = tf.Graph()
#     with g2.as_default():
#         sess = tf.Session()
#         with sess.as_default():
#             print("Loading second saved meta graph...")
#             saver = tf.train.import_meta_graph("{}.meta".format(second_cp))
#             print("Restoring variables...")
#             saver.restore(sess, second_cp)
#             print("Finished TF graph load 2.0.")
#
#             input_x = g2.get_operation_by_name("input_x").outputs[0]
#             input_y = g2.get_operation_by_name("input_y").outputs[0]
#             dropout_keep_prob = g2.get_operation_by_name("dropout_keep_prob").outputs[0]
#
#             predictions = g2.get_operation_by_name("output/predictions").outputs[0]
#             accuracy = g2.get_operation_by_name('accuracy/accuracy').outputs[0]
#
#             acc2, predictions_out2 = sess.run(
#                 [accuracy, predictions],
#                 feed_dict={
#                     input_x: validation_data_2,
#                     input_y: validation_data_check_2,
#                     dropout_keep_prob: 1.0,
#                 })
#
#             print("Second acc:")
#             print(acc)
#
#             print("Computing aggregate predictions now...")
#             pos_res = (predictions_out[:, 0] + predictions_out2[:, 0]) / 2
#             neg_res = (predictions_out[:, 1] + predictions_out2[:, 1]) / 2
#
#             agg = np.vstack([pos_res, neg_res]).T
#             print(agg)
#
#             zero_preds = 0
#             correct = 0
#             for id, pred in enumerate(agg):
#                 if pred[0] >= pred[1]:
#                     zero_preds += 1
#
#                 if (pred[0] >= pred[1] and validation_data_check[id][0] == 1) \
#                         or (pred[0] < pred[1] and validation_data_check[id][1] == 1):
#                     correct += 1
#
#             aggregate_acc = correct * 1.0 / evlim
#             print("Final aggregate accuracy: {0}".format(aggregate_acc))
#
#
#
#
#
#

