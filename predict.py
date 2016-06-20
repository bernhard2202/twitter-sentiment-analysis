import tensorflow as tf
import numpy as np

import pickle
# Used for reliably getting the current hostname.
import socket
import time


tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 1)")

# Example: './data/runs/euler/local-w2v-275d-1466050948/checkpoints/model-96690'
tf.flags.DEFINE_string("checkpoint_file", None, "Checkpoint file from the"
                                                " training run.")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# print("Loading data...")
# with open('./data/preprocessing/vocab.pkl', 'rb') as f:
#     vocabulary = pickle.load(f)
# with open('./data/preprocessing/vocab.pkl', 'rb') as f:
#     vocabulary_inv = pickle.load(f)
# print("Vocabulary size: {:d}".format(len(vocabulary)))

prefix = 'full'
validation_data = np.load('./data/preprocessing/{0}-validateX.npy'.format(prefix))
if FLAGS.checkpoint_file is None:
    raise ValueError("Please specify a TensorFlow checkpoint file to use for"
                     " making the predictions (--checkpoint_file <file>).")

checkpoint_file = FLAGS.checkpoint_file
timestamp = int(time.time())
filename = "./data/output/prediction_cnn_{0}.csv".format(timestamp)
meta_filename = "{0}.meta".format(filename)
print("Predicting using checkpoint file [{0}].".format(checkpoint_file))
print("Will write predictions to file [{0}].".format(filename))

graph = tf.Graph()
with graph.as_default():
    # TODO(andrei): Is this config (and its associated flags) really necessary?
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Collect the predictions here
        all_predictions = []
        id = 1
        for row in validation_data:
            if id % 100 == 0:
                print("done tweets: {:d}".format(id))
            prediction = sess.run(predictions, {input_x: [row], dropout_keep_prob: 1.0})[0]
            all_predictions.append((id, prediction))
            id += 1

        print("Prediction done")
        print("Writing predictions to file...")
        submission = open(filename, 'w+')
        print('Id,Prediction', file=submission)
        for id, pred in all_predictions:
            if pred[0] >= 0.5:
                print("%d,-1" % id,file=submission)
            else:
                print("%d,1" % id,file=submission)

        with open(meta_filename, 'w') as mf:
            print("Generated from checkpoint: {0}".format(checkpoint_file), file=mf)
            print("Hostname: {0}".format(socket.gethostname()), file=mf)

        print("...done.")
        print("Wrote predictions to: {0}".format(filename))
        print("Wrote some simple metadata about how the predictions were"
              " generated to: {0}".format(meta_filename))
