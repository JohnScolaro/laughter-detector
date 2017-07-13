import tensorflow as tf
import numpy as np
import matplotlib as mpl
import os
import helpers
import time
import sys

start_time = time.time()
directory = os.path.dirname(__file__)
save_folder_name = helpers.get_save_dir(os.path.join(directory, 'tensorboard'), "mlp_test")
pics_save_path = os.path.join(save_folder_name, 'pics')
log_path = os.path.join(save_folder_name, 'logs')

# Make lists of valid dataset file names
dataset_file_list = []
for x in range(100):
    name = os.path.join(directory, '../../Dataset/') + str(x) + "_dataset.tfrecord"
    if os.path.isfile(name):
        dataset_file_list.append(name)

# Start up a logger to log misc variables.
logger = helpers.Logger(save_folder_name)

# System Parameters
learning_rate = 0.001
training_epochs = 30
display_step = 50
batch_size = 5000
train_test_ratio = 0.85
n_input = 60 # Data input features
n_classes = 2 # Output types. Either laughter or not laughter.
lines_in_one_epoch = helpers.total_lines_in_all_tfrecord_files(
        dataset_file_list) * train_test_ratio

# Construct input pipelines
train_iter, test_iter = helpers.input_pipeline2(dataset_file_list, batch_size)
data, clip, label = train_iter.get_next()
test_data, test_clip, test_label = test_iter.get_next()

# Construct model
mlp_train, mlp_test = helpers.multilayer_perceptron(data, test_data, n_input, n_classes, [200])

# Define cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mlp_train,
        labels=label), name="cost_op")
tf.summary.scalar('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
        name='Adam_Optimizer').minimize(cost)

# Collect all summaries into one handle and create a writer
merged_summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter(save_folder_name)

# Create handles for accuracy and confusion calculation
test_op, reset_op, accuracy, confusion = helpers.streaming_accuracy_and_confusion_calculation(
        test_label, mlp_test, n_classes)

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
                helpers.save_confusion_matrix(conf, pics_save_path,
                        classes=['Not Laughter', 'Laughter'],
                        name='confusion_at_epoch_' + str(cur_epoch_num))
                sens, spec = helpers.sensitivity_and_specificity(conf)
                print("Sensitivity: {:.3f}, Specificity: {:.3f}".format(sens, spec))
                logger.log_scalar("Sensitivity", sens, cur_epoch_num)
                logger.log_scalar("Specificity", spec, cur_epoch_num)

                # Reset streaming metrics
                sess.run(reset_op)
                break

    # Now do all the end of training testing specific operations.
    print("Training Completed in {:.3f} seconds.".format(time.time() - start_time))

    # Save confusion matrices
    helpers.save_confusion_matrix(conf, pics_save_path,
            classes=['Not Laughter', 'Laughter'],
            name='final_confusion_matrix')
    helpers.save_confusion_matrix(conf, pics_save_path,
            classes=['Not Laughter', 'Laughter'], normalize=True,
            name='final_norm_confusion_matrix')

# Finally, open our TensorBoard tab
helpers.openTensorBoard(save_folder_name)
