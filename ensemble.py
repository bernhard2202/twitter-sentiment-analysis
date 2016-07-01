"""Utility for computing simple averaged predictions.

Also supports evaluating simple ensembles of two NNs.
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


# This ensures our predictions are consistent.
random_seed = 0x123
random.seed(random_seed)
np.random.seed(random_seed)

# default_pp = os.path.join('data', 'preprocessing')
default_pp = os.path.join('data', 'preprocessing-bogus')

# Example: './data/runs/euler/local-w2v-275d-1466050948/checkpoints/model-96690'
tf.flags.DEFINE_string("checkpoint_file", None, "Checkpoint file from the"
                                                " training run.")
tf.flags.DEFINE_string(
    "second_checkpoint_file",
    None,
    "Another checkpoint file for computing avg. probabilities. If this is"
    " provided, a second config is checked and the train accuracy for this"
    " checkpoint is also verified. Specifying this also triggers the"
    " computation of the 'ensemble' predictions, done with probability"
    " averaging.")
tf.flags.DEFINE_string("validation_data_file",
                       os.path.join(default_pp, 'validateX.npy'),
                       "Data used for Kaggle prediction using ensemble, if"
                       " enabled.")
tf.flags.DEFINE_bool("train_error", False, "Whether to actually validate"
                                           " models on the training set.")
FLAGS = tf.flags.FLAGS

# How many data points to evaluate out of the training data.
# Does not affect the number of total predictions generated for Kaggle if the
# second checkpoint file is provided.
evlim = 10000


class ModelConfig(object):
    """Specifies the input files and placeholder names for a checkpoint.

    Useful for checkpoints with inconsistent names trained on data preprocessed
    in different ways.
    """

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

        if FLAGS.train_error:
            self.trainx = np.load(trainx_fname)
            self.trainy = np.load(trainy_fname)

            # Shuffle the data in a predictable way, so that multiple configs
            # load different data files (preprocessed differently) but shuffle them
            # in an identical way.
            self.trainx, self.trainy = shuffle(self.trainx, self.trainy,
                                               random_state=random_seed)
            self.trainx = self.trainx[:evlim]
            self.trainy = self.trainy[:evlim]

        self.validx = np.load(valid_fname)


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


# TODO(andrei): Remove code duplication between this and 'evaluate'.
def predict(config: ModelConfig) -> np.ndarray:
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

        if config.dropout_name is not None:
            dropout_keep_prob = graph.get_operation_by_name(config.dropout_name).outputs[0]
        else:
            dropout_keep_prob = None

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name(config.predictions_name).outputs[0]

        feed_dict = {input_x: config.validx}
        if dropout_keep_prob is not None:
            feed_dict[dropout_keep_prob] = 1.0

        predictions_out, = sess.run([predictions], feed_dict)
        return predictions_out


def main(_):
    if FLAGS.checkpoint_file is None:
        raise ValueError(
            "Please specify a TensorFlow checkpoint file to use for making the"
            " predictions (--checkpoint_file <file>).")

    # TODO(andrei): Document better or scrap in favor of pre-written configs.
    if FLAGS.second_checkpoint_file is not None:
        print("Using poor man's ensemble.")
        second_cp = FLAGS.second_checkpoint_file
    else:
        second_cp = None

    print("Will shuffle data data and limit it to {0} points.".format(evlim))

    # These configurations need to be adjusted according to your local
    # preprocessed files. This is more effective than having to set 10+ flags.
    config_A = ModelConfig(
        FLAGS.checkpoint_file,
        # trainx_fname='data/preprocessing-old-lstm/full-trainX.npy',
        # trainy_fname='data/preprocessing-old-lstm/full-trainY.npy',
        valid_fname='data/key-checkpoints/validateXlstm.npy',
        input_x_name='input_batch_x',
        input_y_name='input_batch_x_1',
        predictions_name='output/Softmax',
        accuracies_name='accuracy/Mean')

    if FLAGS.train_error:
        acc_A, preds_A = evaluate(config_A)
        print("Accuracy A using clean mode: {0}".format(acc_A))

    if second_cp is not None:
        config_B = ModelConfig(second_cp,
                               valid_fname='data/key-checkpoints/validateXcnn.npy')

        if FLAGS.train_error:
            acc_B, preds_B = evaluate(config_B)
            print("Accuracy B using clean mode: {0}".format(acc_B))

            print("Computing aggregate predictions now...")
            pos_res = (preds_A[:, 0] + preds_B[:, 0]) / 2
            neg_res = (preds_A[:, 1] + preds_B[:, 1]) / 2

            agg = np.vstack([pos_res, neg_res]).T
            print(agg)

            assert 0 == np.sum(~np.equal(config_A.trainy, config_B.trainy)), \
                "There should be no discrepancies between the training labels of" \
                " the two configs, since they are not preprocessed."

        if FLAGS.train_error:
            train_y = config_A.trainy
            zero_preds = 0
            correct = 0
            for id, pred in enumerate(agg):
                if pred[0] >= pred[1]:
                    zero_preds += 1

                if (pred[0] >= 0.5 and train_y[id][0] == 1) \
                        or (pred[0] < 0.5 and train_y[id][1] == 1):
                    correct += 1

            aggregate_acc = correct * 1.0 / evlim
            print("Final aggregate accuracy: {0}".format(aggregate_acc))

        # TODO(andrei): Make code a little less wasteful in terms of reloading
        # models.
        print("Will now perform official prediction using ensemble.")
        preds_A = predict(config_A)
        preds_B = predict(config_B)
        pos_res = (preds_A[:, 0] + preds_B[:, 0]) / 2
        neg_res = (preds_A[:, 1] + preds_B[:, 1]) / 2
        agg = np.vstack([pos_res, neg_res]).T

        timestamp = int(time.time())
        ensemble_out_filename = "./data/output/prediction_ensemble_{0}.csv".format(timestamp)
        with open(ensemble_out_filename, 'w') as f:
            f.write("Id,Prediction\n")
            for id, pred in enumerate(agg):
                if pred[0] >= 0.5:
                    f.write("{0},-1\n".format(id + 1))
                else:
                    f.write("{0},1\n".format(id + 1))

        print("Done writing predictions to file {0}.".format(
            ensemble_out_filename))


if __name__ == '__main__':
    tf.app.run(main)
