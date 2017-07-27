import tensorflow as tf
import numpy as np
import matplotlib as mpl
import os
import helpers
import plotters
import time
import sys

################################################################################

# If running this file:
if len(sys.argv) < 2:
    name = "sequence_mlp_test"

    # Hyper Parameters
    learning_rate = 0.001 #0.001
    beta1 = 0.9 #0.9
    beta2 = 0.999 #0.999
    epsilon = 1e-08 #1e-08

    # Network Params
    training_epochs = 1
    display_step = 50
    batch_size = 5000
    train_test_ratio = 0.85
    activation_function = 'relu'
    layers = [400]
    output_layer_biases = True
    n_input = 60 # Data input features
    n_classes = 2 # Output types. Either laughter or not laughter.
    window_length = 20

################################################################################

# Else, if running from script:
else:
    name = sys.argv[1]
    learning_rate = float(sys.argv[2])
    beta1 = float(sys.argv[3])
    beta2 = float(sys.argv[4])
    epsilon = float(sys.argv[5])
    training_epochs = int(sys.argv[6])
    display_step = int(sys.argv[7])
    batch_size = int(sys.argv[8])
    train_test_ratio = float(sys.argv[9])
    activation_function = sys.argv[10]
    layers = sys.argv[11].strip('[]').split('-')
    for x in range(len(layers)):
        layers[x] = int(layers[x])
    output_layer_biases = "True"== sys.argv[12]
    n_input = int(sys.argv[13])
    n_classes = int(sys.argv[14])

start_time = time.time()
directory = os.path.dirname(__file__)
save_folder_name = helpers.get_save_dir(os.path.join(directory, 'tensorboard'), name)
pics_save_path = os.path.join(save_folder_name, 'pics')
log_save_path = os.path.join(save_folder_name, 'log')

# Close any other local sessions if there are any.
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

# If running from another file, redirect output to a log file.
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)
if not os.path.isdir(log_save_path):
    os.makedirs(log_save_path)
if len(sys.argv) > 1:
    stderr = sys.stderr
    stdout = sys.stdout
    sys.stderr = sys.stdout = open(os.path.join(log_save_path, 'log.txt'), 'w')

# Make lists of valid dataset file names
dataset_file_list = []
for x in range(100):
    name = os.path.join(directory, '../Dataset/') + str(x) + "_dataset.tfrecord"
    if os.path.isfile(name):
        dataset_file_list.append(name)

# Print system params
if len(sys.argv) > 2:
    helpers.print_system_params(sys.argv)

# Construct input pipelines
train_iter, test_iter = helpers.input_pipeline2(dataset_file_list, batch_size)
data, clip, label = train_iter.get_next()
test_data, test_clip, test_label = test_iter.get_next()

# Add to pipeline in order to batch sequences
data, label = helpers.input_pipeline_data_sequence_creator(data, label,
        batch_size, window_length, n_input, n_classes)
test_data, test_label = helpers.input_pipeline_data_sequence_creator(test_data,
        test_label, batch_size, window_length, n_input, n_classes)

# Construct model
mlp_train, mlp_test = helpers.sequence_mlp(data, test_data, n_input,
        window_length, n_classes, batch_size, layers,
        activation_function=activation_function,
        output_layer_biases=output_layer_biases)

# Define cost and optimizer
weighted_labels = tf.multiply(label, tf.constant([1, 25], dtype=tf.int32), name='add_weight_to_labels')
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mlp_train,
        labels=weighted_labels, name="cost_op"))
tf.summary.scalar('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1,
        beta2=beta2, epsilon=epsilon, name='Adam_Optimizer').minimize(cost)

# Collect all summaries into one handle and create a writer
merged_summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter(save_folder_name)

# Start up a logger to log misc variables.
logger = helpers.Logger(save_folder_name, writer)
metrics = helpers.Metrics()

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

    # Write the graph of the system to disk so we can view it later.
    writer.add_graph(sess.graph)

    # Training loop
    tot_batch = 0
    for cur_epoch_num in range(training_epochs):

        batch = 0

        # Reinitialize input dataset
        sess.run(train_iter.initializer)
        sess.run(test_iter.initializer)

        while 1:

            # Try to fetch a new batch. If there are none left, we are done.
            try:

                if batch % display_step != 0:

                    # Run optimization op (backprop) and cost op.
                    sess.run([optimizer, cost])

                else:

                    # Run optimization op (backprop) and cost op with more info.
                    _, c, s = sess.run([optimizer, cost, merged_summaries],
                            options=run_options, run_metadata=run_metadata)

                    # Display some info about the current training session.
                    last_time = cur_time
                    cur_time = time.time()
                    print("Batch number = {:d}. Epoch {:d}. Batch Cost: {:.3f}. {:.3f}s per batch. Total Time = {:.3f}s.".format(batch,
                            cur_epoch_num, c, (cur_time - last_time) /
                            display_step, cur_time - start_time))

                    # Write data to summary's on each display batch
                    writer.add_run_metadata(run_metadata, 'batch' + str(tot_batch))
                    writer.add_summary(s, tot_batch)

            except tf.errors.OutOfRangeError:
                print("Finished Epoch {:d}.".format(cur_epoch_num))
                break

            batch = batch + 1
            tot_batch = tot_batch + 1

        # Now do all the "end of epoch" testing.
        while 1:
            try:
                sess.run(test_op)
            except tf.errors.OutOfRangeError:
                acc, conf = sess.run([accuracy, confusion])
                print("Accuracy: {:.5f}".format(acc))
                print("Confusion Matrix:")
                print(conf)
                logger.log_scalar("Test_Accuracy", acc, cur_epoch_num)
                plotters.save_confusion_matrix(conf, pics_save_path,
                        classes=['Not Laughter', 'Laughter'],
                        name='confusion_at_epoch_' + str(cur_epoch_num))
                sens, spec = metrics.sensitivity_and_specificity(conf)
                print("Sensitivity: {:.3f}, Specificity: {:.3f}".format(sens, spec))
                logger.log_scalar("Sensitivity", sens, cur_epoch_num)
                logger.log_scalar("Specificity", spec, cur_epoch_num)

                # Reset streaming metrics
                sess.run(reset_op)
                break

        # Re-initialize the test data, so you can check again, to draw pics.
        sess.run(test_iter.initializer)
        for x in range(50):

            lab, pred = sess.run([test_label, soft_mlp_test])
            plotters.laughter_plotter(pred, lab, pics_save_path, x, 0.02,
                    batch_size)

    # Now do all the end of training testing specific operations.
    print("Training Completed in {:.3f} seconds.".format(time.time() - start_time))

    sens, spec, epoch = metrics.get_max_sensitivity_and_specificity()
    print("Best test run was epoch {:d} of {:d} with a sensitivity of {:.3f}, and specificity of {:.3f}.".format(epoch + 1, training_epochs, sens, spec))

    # Save confusion matrices
    plotters.save_confusion_matrix(conf, pics_save_path,
            classes=['Not Laughter', 'Laughter'],
            name='final_confusion_matrix')
    plotters.save_confusion_matrix(conf, pics_save_path,
            classes=['Not Laughter', 'Laughter'], normalize=True,
            name='final_norm_confusion_matrix')

# Finally, open our TensorBoard tab (If not being called from net_runner)
if len(sys.argv) < 2:
    helpers.openTensorBoard(save_folder_name)
else:
    sys.stdout.close()
    sys.stdout = stdout
    sys.stderr = stderr
