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

import random
import time


random.seed(0x123)
np.random.seed(0x123)

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

# validation_data_fname = './data/preprocessing/validateX.npy'

# validation_data_fname = './data/preprocessing/subset-trainX.npy'
# validation_data_check_fname = './data/preprocessing/subset-trainY.npy'

# validation_data_fname = './data/preprocessing/full-trainX.npy'
# validation_data_check_fname = './data/preprocessing/full-trainY.npy'

# TODO(andrei): Flagify this.
validation_data_fname = '/tmp/cil/full-trainX.npy'
validation_data_check_fname = '/tmp/cil/full-trainY.npy'

validation_data = np.load(validation_data_fname)
validation_data_check = np.load(validation_data_check_fname)

# TODO(andrei): Clean this shit up...
validation_data_2 = np.load('data/preprocessing/full-trainX.npy')
validation_data_check_2 = np.load('data/preprocessing/full-trainY.npy')

# This bit does subsampling of validation data, when performing sanity check
# by recomputing train error.

# How many data points to evaluate.
evlim = 50

validation_data, validation_data_check, validation_data_2, validation_data_check_2 = shuffle(validation_data, validation_data_check, validation_data_2, validation_data_check_2)
validation_data = validation_data[:evlim]
validation_data_check = validation_data_check[:evlim]
validation_data_2 = validation_data_2[:evlim]
validation_data_check_2 = validation_data_check_2[:evlim]

print("Shuffled data and limited to {0} points.".format(evlim))
cat_counts = np.sum(validation_data_check, axis=0)
print("Label counts (pos/neg): {0}".format(cat_counts))

if FLAGS.checkpoint_file is None:
    raise ValueError("Please specify a TensorFlow checkpoint file to use for"
                     " making the predictions (--checkpoint_file <file>).")

checkpoint_file = FLAGS.checkpoint_file
timestamp = int(time.time())

print("Evaluating model from checkpoint file [{0}].".format(checkpoint_file))
print("Validation data shape: {0}".format(validation_data.shape))


class ModelProxy(object):

    def __init__(self, model, ix_name, dropout_name, predictions_name):
        self.model = model
        self.ix_name = ix_name

        self.input_x = model.get_operation_by_name(ix_name).outputs[0]
        if dropout_name is not None:
            self.dropout_keep_prob = model.get_operation_by_name(dropout_name).outputs[0]
        else:
            self.dropout_keep_prob = None

        self.predictions = model.get_operation_by_name(predictions_name).outputs[0]

    def predict(self, session):
        feed_dict = {
            self.input_x: [row],
        }
        if self.dropout_keep_prob is not None:
            feed_dict[self.dropout_keep_prob] = 1.0

        return session.run(self.predictions, feed_dict)[0]


def evaluate_ensemble(model_proxies):
    pass


graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        print("Loading saved meta graph...")
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        print("Restoring variables...")
        saver.restore(sess, checkpoint_file)
        print("Finished TF graph load.")

        if second_cp is not None:
            print("Loading second checkpoint.")

        # aux = [o.name for o in graph.get_operations() if 'input' in o.name]
        # print("Interesting ops:")
        # print(aux)
        # print()

        # Get the placeholders from the graph by name
        # If you forget to name your input, try 'Placeholder', or 'Placeholder_1'.
        # input_x = graph.get_operation_by_name("Placeholder").outputs[0]
        # input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]

        input_x = graph.get_operation_by_name("input_batch_x").outputs[0]
        input_y = graph.get_operation_by_name("input_batch_x_1").outputs[0]

        # input_y = graph.get_operation_by_name("Placeholder_1").outputs[0]
        # input_y = graph.get_operation_by_name("input_batch_x_1").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/Softmax").outputs[0]
        # predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        accuracy = graph.get_operation_by_name('accuracy/Mean').outputs[0]
        # accuracy = graph.get_operation_by_name('accuracy/accuracy').outputs[0]

        acc, predictions_out = sess.run(
            [accuracy, predictions],
            feed_dict={
                input_x: validation_data,
                input_y: validation_data_check,
                dropout_keep_prob: 1.0,
            })
        print("Official accuracy:")
        print(acc)

        print("Official predictions we made:")
        print(predictions_out)
        # print("Exiting early.")
        # exit(0)

        # Collect the predictions here
        # TODO(andrei): Nice error bar because you have nothing better to do you
        # worthless lazy procrastinating sack of shit.
        # TODO(andrei): Fix self-esteem issues.
        all_predictions = []
        print("Computing accuracy in the old-fashioned way...")
        print("Consider interrupting execution unless you're paranoid.")
        try:
            for (id, row) in enumerate(validation_data):
                if (id + 1) % 1000 == 0:
                    print("Done tweets: {0}/{1}".format(id + 1, len(validation_data)))

                # XXX: Debug
                if id > evlim:
                    break

                # TODO(andrei): Why does running 'predictions' return TWO identical rows?
                prediction = sess.run(predictions, {
                    input_x: [row],
                    dropout_keep_prob: 1.0
                })[0]
                all_predictions.append((id, prediction))

            print("Prediction done")
            correct = 0

            print(all_predictions[0])
            print(all_predictions[1])

            # Keep track of maybe a weird bias?
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
            print("Naive accuracy (sanity check):")
            print(acc)

            print("We had {0} cases with pred[0] >= pred[1] out of {1}.".format(
                zero_preds, evlim))
        except KeyboardInterrupt:
            print("K, nvm.")

        print("Finished evaluation of model from checkpoint file {0}.".format(
            checkpoint_file))

if second_cp is not None:
    g2 = tf.Graph()
    with g2.as_default():
        sess = tf.Session()
        with sess.as_default():
            print("Loading second saved meta graph...")
            saver = tf.train.import_meta_graph("{}.meta".format(second_cp))
            print("Restoring variables...")
            saver.restore(sess, second_cp)
            print("Finished TF graph load 2.0.")

            input_x = g2.get_operation_by_name("input_x").outputs[0]
            input_y = g2.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = g2.get_operation_by_name("dropout_keep_prob").outputs[0]

            predictions = g2.get_operation_by_name("output/predictions").outputs[0]
            accuracy = g2.get_operation_by_name('accuracy/accuracy').outputs[0]

            acc, predictions_out = sess.run(
                [accuracy, predictions],
                feed_dict={
                    input_x: validation_data_2,
                    input_y: validation_data_check_2,
                    dropout_keep_prob: 1.0,
                })

            print("Second acc:")
            print(acc)

