import tensorflow as tf
import numpy as np
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import plotters
import helpers

name = "demo"

# Network Params
batch_size = 200
activation_function = 'relu'
layers = [400]
output_layer_biases = True
n_input = 20 # Data input features
n_classes = 2 # Output types. Either laughter or not laughter.
window_length = 50

# Creating paths
start_time = time.time()
directory = os.path.dirname(__file__)
save_folder_name = helpers.get_save_dir(os.path.join(directory, 'output'), name)
pics_save_path = os.path.join(save_folder_name, 'pics')
log_save_path = os.path.join(save_folder_name, 'log')
save_save_path = os.path.join(save_folder_name, 'weights')
restore_path = os.path.join(directory, '..', 'tensorboard', 'sequence_mlp_test65', 'weights', 'data')

# Close any other local sessions if there are any.
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

dataset_file_list = []
dataset_file_list.append(os.path.join(directory, 'audio_file', '1_dataset.tfrecord'))

# Construct input pipelines
test_iter = helpers.input_pipeline3_demo(dataset_file_list, window_length,
        n_classes, batch_size, n_input, batch_normalize=True)
test_data, test_clip, test_seq, test_label = test_iter.get_next()

# Construct model loading saved data
mlp_train, mlp_test, weights, biases = helpers.sequence_mlp(test_data, test_data,
        n_input, window_length, n_classes, batch_size, layers,
        activation_function=activation_function,
        output_layer_biases=output_layer_biases,
        initialization='Xavier', clipped=False, dropout=False)

# Collect all summaries into one handle and create a writer
merged_summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter(save_folder_name)

# Start up a logger to log misc variables.
logger = helpers.Logger(save_folder_name, writer)
metrics = helpers.Metrics()

# Create a saver to load variables later.
saver = tf.train.Saver()

# Create handles for accuracy and confusion calculation
test_op, reset_op, accuracy, confusion = helpers.streaming_accuracy_and_confusion_calculation(
        test_label, mlp_test, n_classes)

# Create handles for probability visualisation.
soft_mlp_test = tf.nn.softmax(mlp_test, name='test_softmax')

# Collect metadata about the train. Calc times, memory used, device, etc.
run_metadata = tf.RunMetadata()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

# Initializing local and global variables
init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())

# Print time to launch main session
cur_time = time.time()
print("Launching TensorFlow Session after {:.3f} seconds.".format(cur_time -
        start_time))

# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    # Finalize the graph so no more changes can be made
    sess.graph.finalize()

    # Initialize all system variables
    sess.run(init_op)
    saver.restore(sess, restore_path)

    # Initialize dataset iterator
    sess.run(test_iter.initializer)

    # Do all testing below here:

    # Get the final metrics from the system
    conf = metrics.end_of_epoch_sens_spec(sess, accuracy, confusion,
            test_op, reset_op, logger, 0,
            pics_save_path)

    # Plot the laughter classification plots.
    plotters.multiple_laughter_plotter(sess, test_label, test_clip, test_seq,
            soft_mlp_test, test_iter, pics_save_path, batch_size, window_length)

    # Plot the ROC curves.
    plotters.multiple_roc_curve_plotter(sess, test_label, soft_mlp_test,
            test_iter, pics_save_path)

    # Print useful information
    print("Testing Completed in {:.3f} seconds.".format(time.time() - start_time))

    # Save final confusion matrices
    plotters.save_final_confusion_matrixes(conf, pics_save_path)
