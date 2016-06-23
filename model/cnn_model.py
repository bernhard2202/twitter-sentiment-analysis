import tensorflow as tf


class TextCNN(object):
    """A CNN for text classification.

    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """

        :param sequence_length: tweet word length
        :param num_classes: number of classes for classification
        :param vocab_size: vocabulary size
        :param embedding_size: dimensions of embeddings
        :param filter_sizes: filter size (number of words convolutions should cover)
        :param num_filters: number of filters to use
        :param l2_reg_lambda: l2 regularization constant lambda

        """

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            self.embeddings = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="embeddings")

            self.embedded_words = tf.nn.embedding_lookup(self.embeddings, self.input_x)
            self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)

            # code for static embeddings:
            #self.embeddings_static = tf.Variable(
            #    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            #    name="embeddings.static", trainable=False)
            #self.embedded_words_static = tf.nn.embedding_lookup(self.embeddings_static, self.input_x)
            #self.embedded_words_expanded_static = tf.expand_dims(self.embedded_words_static, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                filter_matrix = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_matrix")
                bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bias")
                convolutions = tf.nn.conv2d(
                    self.embedded_words_expanded, # [batch, in_height, in_width, in_channels]
                    filter_matrix, # [filter_height, filter_width, in_channels, out_channels]
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(convolutions, bias), name="relu")
                # Maxpooling over the outputs
                # leaves us with a tensor of shape [batch_size, 1, 1, num_filters]
                pooled = tf.nn.max_pool(
                    h,  # [batch, height, width, channels]
                    # size of the window for each dimension of the input tensor.
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],  # stride of the sliding window for each dimension of the input tensor
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

                # Convolutions for static embeddings:

                #filter_matrix_s = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_matrix")
                #bias_s = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bias")
                #conv_s = tf.nn.conv2d(
                #    self.embedded_words_expanded_static, # [batch, in_height, in_width, in_channels]
                #    filter_matrix_s, # [filter_height, filter_width, in_channels, out_channels]
                #    strides=[1, 1, 1, 1],
                #    padding="VALID",
                #    name="conv_s")
                # Apply nonlinearity
                #h_s = tf.nn.relu(tf.nn.bias_add(conv_s, bias_s), name="relu")
                # Maxpooling over the outputs
                # leaves us with a tensor of shape [batch_size, 1, 1, num_filters]
                #pooled_s = tf.nn.max_pool(
                #    h_s, #[batch, height, width, channels]
                #    # size of the window for each dimension of the input tensor.
                #    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                #    strides=[1, 1, 1, 1], # stride of the sliding window for each dimension of the input tensor
                #    padding='VALID',
                #    name="pool_s")
                #pooled_outputs.append(pooled_s)

        # Combine all the pooled features
        # create a vecotr of shape [batch_size, num_filters_total]
        num_filters_total = num_filters * len(filter_sizes) # *2
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            output_weights = tf.get_variable(
                "output_weights",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(output_weights)
            l2_loss += tf.nn.l2_loss(bias)
            self.scores = tf.nn.xw_plus_b(self.h_drop, output_weights, bias, name="scores")

            self.predictions = tf.nn.softmax(self.scores, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.predictions,1), tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")