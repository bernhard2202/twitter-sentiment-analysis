"""LSTM (Long Short-Term Memory) NN for tweet sentiment analysis."""

import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell, rnn


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
    TODO(andrei): Consider using 'EmbeddingWrapper'.
    TODO(andrei): NEVER use 'tf.Variable' directly!
    TODO(andrei): Use 'variable_scope' instead of 'name_scope'.
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
        # Want:         [(batch_size, n_hidden) * n_words]
        #
        # Since otherwise there's no way to feed information into the LSTM cell.
        # Yes, it's a bit confusing, because we want a batch of multiple
        # sequences, with each step being of 'embedding_size'.
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
    def _cell(input_size, num_units, use_peepholes, dropout_keep_prob):
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
