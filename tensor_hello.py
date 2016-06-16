"""Helper program for testing whether TensorFlow works."""

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
session = tf.Session()
print(session.run(hello))
