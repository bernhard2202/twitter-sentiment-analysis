"""LSTM (Long Short-Term Memory) NN for tweet sentiment analysis."""

import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell, rnn

from .util import batch_iter


class TextLSTM(object):
    """Simple LSTM for binary text classification using word embeddings.

    Args:
        sequence_length: The length of a Tweet. Current architecture requires
                         this, but could be extended to not care.
        vocab_size: vocabulary size
        embedding_size: dimensions of embeddings
        hidden_size: The size of the hidden state (and memory).
        layer_count: The number of LSTM layers. Must be >=1. When just 1 is
                     given, it will be the size of 'hidden_size', even if the
                     embedding size may be different. If it's >1, then the
                     first cell layer is the same dimension as the embeddings,
                     and subsequent layers are all 'hidden_size'-dimensional.

    TODO(andrei): Also try a bidirectional RNN.
    TODO(andrei): Experiment with deeper LSTM.
    TODO(andrei): Consider using 'EmbeddingWrapper'.
    """

    def __init__(self, sequence_length, vocab_size, embedding_size, hidden_size,
                 layer_count=1, **kw):
        assert layer_count >= 1, "An LSTM cannot have less than one layer."
        n_classes = kw.get('n_classes', 2)  # >2 not tested.
        self.input_x = tf.placeholder(tf.int32,
                                      [None, sequence_length],
                                      name="input_x")
        self.input_y = tf.placeholder(tf.float32,
                                      [None, n_classes],
                                      name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                name="dropout_keep_prob")

        # Layer 1: Word embeddings
        self.embeddings = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1),
            name="embeddings")
        embedded_words = tf.nn.embedding_lookup(self.embeddings, self.input_x)

        # Funnel the words into the LSTM.
        # Current size: (batch_size, n_words, emb_dim)
        # Want:         [(batch_size, n_hidden) * n_words <- ??? IS THIS RIGHT?]
        #
        # Since otherwise there's no way to feed information into the LSTM cell.
        embedded_words = tf.transpose(embedded_words, [1, 0, 2])
        embedded_words = tf.reshape(embedded_words, [-1, embedding_size])
        # Note: 'tf.split' outputs a **Python** list.
        embedded_words = tf.split(0, sequence_length, embedded_words)

        # Layer 2: LSTM cell
        lstm_use_peepholes = True
        # 'state_is_tuple = True' should NOT be used despite the warnings
        # (which appear as of TF 0.9), since it doesn't work on the version of
        # TF installed on Euler (0.8).
        if layer_count > 1:
            print("Using deep {0}-layer LSTM with first layer size {1}"
                  " (embedding size) and hidden layer size {2}."
                  .format(layer_count, embedding_size, hidden_size))
            print("First cell {0}->{1}".format(embedding_size, embedding_size))
            first_cell = TextLSTM._cell(embedding_size,
                                        embedding_size,
                                        lstm_use_peepholes,
                                        self.dropout_keep_prob)
            print("Second cell {0}->{1}".format(embedding_size, hidden_size))
            second_cell = TextLSTM._cell(embedding_size,
                                         hidden_size,
                                         lstm_use_peepholes,
                                         self.dropout_keep_prob)
            print("Third cell+ {0}->{1} (if applicable)".format(hidden_size,
                                                                hidden_size))
            third_plus = TextLSTM._cell(hidden_size,
                                        hidden_size,
                                        lstm_use_peepholes,
                                        self.dropout_keep_prob)
            deep_cells = [third_plus] * (layer_count - 2)
            lstm_cells = rnn_cell.MultiRNNCell([first_cell, second_cell] +
                                               deep_cells)
        else:
            print("Using simple 1-layer LSTM with hidden layer size {0}."
                  .format(hidden_size))
            lstm_cells = rnn_cell.LSTMCell(num_units=hidden_size,
                                           input_size=embedding_size,
                                           forget_bias=1.0,
                                           use_peepholes=lstm_use_peepholes)

        # Q: Can't batches end up containing both positive and negative labels?
        #    Can the LSTM batch training deal with this?
        #
        # A: Yes. Each batch feeds each sentence into the LSTM, incurs the loss,
        #    and backpropagates the error separately. Each example in a bath
        #    is independent. Note that as opposed to language models, for
        #    instance, where we incur a loss for all outputs, in this case we
        #    only care about the final output of the RNN, since it doesn't make
        #    sense to classify incomplete tweets.

        outputs, _states = rnn(lstm_cells,
                               inputs=embedded_words,
                               dtype=tf.float32)

        # Layer 3: Final Softmax
        out_weight = tf.Variable(tf.random_normal([hidden_size, n_classes]))
        out_bias = tf.Variable(tf.random_normal([n_classes]))

        with tf.name_scope("output"):
            lstm_final_output = outputs[-1]
            self.scores = tf.nn.xw_plus_b(lstm_final_output, out_weight,
                                          out_bias, name="scores")
            self.predictions = tf.nn.softmax(self.scores, name="predictions")

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(self.scores,
                                                                  self.input_y)
            self.loss = tf.reduce_mean(self.losses, name="loss")

        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(tf.argmax(self.predictions, 1),
                                         tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"),
                                           name="accuracy")

    @staticmethod
    def _cell(
        input_size: int,
        num_units: int,
        use_peepholes: bool,
        dropout_keep_prob: int
    ) -> rnn_cell.LSTMCell:
        """Helper for building an LSTM cell."""

        # Note: 'input_size' is deprecated in TF 0.9, but required in TF 0.8
        # since our cells aren't all the same size.
        cell = rnn_cell.LSTMCell(
            num_units=num_units,
            input_size=input_size,
            forget_bias=1.0,
            use_peepholes=use_peepholes)
        dropout_cell = rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=dropout_keep_prob,
            output_keep_prob=dropout_keep_prob)
        return dropout_cell


def solve(x_data, y_data, vocabulary, embeddings):
    """Testing function. For serious runs, please use 'train_model' instead."""
    learning_rate = 1e-4
    batch_size = 256
    display_step = 10
    epochs = 10

    # LSTM parameters
    n_input = 300           # Embedding dimensionality
    n_steps = 35            # Sentence length
    n_hidden = 128          # How complex the internal representation is. Having
    # it the same as the embedding size makes the code easier, but is by no
    # means a requirement or the best possible options.
    layer_count = 3

    lstm = TextLSTM(n_steps, len(vocabulary), n_input, n_hidden,
                    layer_count=layer_count)

    # Input: the sentences, represented as lists of indices.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(lstm.loss)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(lstm.embeddings.assign(embeddings))

        merged_data = np.array(list(zip(x_data, y_data)))
        print("merged data len:", len(merged_data), merged_data.shape)
        training_iters = int(len(merged_data) * epochs)
        batches = batch_iter(merged_data, batch_size, epochs)
        for (i, batch) in enumerate(batches):
            batch_x, batch_y = zip(*batch)

            sess.run(optimizer, feed_dict={lstm.input_x: batch_x,
                                           lstm.input_y: batch_y})
            if (i + 1) % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(lstm.accuracy, feed_dict={lstm.input_x: batch_x,
                                                         lstm.input_y: batch_y})
                # Calculate batch loss
                loss = sess.run(lstm.loss, feed_dict={lstm.input_x: batch_x,
                                                      lstm.input_y: batch_y})
                print("It [{0}/{1}]: MinibatchLoss={2:.6f},"
                      " TrainAcc={3:.5f}".format(
                        i * batch_size,
                        training_iters,
                        loss,
                        acc))


def main():
    data_root = './data'
    pp = os.path.join(data_root, 'preprocessing')
    prefix = 'subset'

    print("Data root is: {0}.".format(data_root))
    print("Actual files in: {0}.".format(pp))
    print("Using input data file prefix: {0}.".format(prefix))

    with open(os.path.join(pp, '{0}-vocab.pkl'.format(prefix)), 'rb') as f:
        vocabulary = pickle.load(f)

    x = np.load(os.path.join(pp, '{0}-trainX.npy'.format(prefix)))
    y = np.load(os.path.join(pp, '{0}-trainY.npy'.format(prefix)))
    embeddings = np.load(os.path.join(pp, '{0}-embeddings.npy'.format(prefix)))

    print("Finished loading everything.")
    print("Input shape: {0}".format(x.shape))

    solve(x, y, vocabulary, embeddings)


if __name__ == '__main__':
    main()
