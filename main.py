import tensorflow as tf
import numpy as np
import matplotlib as mpl
import os
import helpers
import time

start_time = time.time()
directory = os.path.dirname(__file__)
save_folder_name = helpers.get_save_dir(os.path.join(directory, 'tensorboard'), "mlp_test")

# Make lists of valid dataset file names
dataset_file_list = []
for x in range(100):
    name = os.path.join(directory, '../../Dataset/') + str(x) + "_dataset.tfrecord"
    if os.path.isfile(name):
        dataset_file_list.append(name)

# Start up a logger to log misc variables.
logger = helpers.Logger(save_folder_name)

# System Parameters
learning_rate = 0.5
training_epochs = 1
display_step = 10
batch_size = 5000
train_test_ratio = 0.85
lines_in_one_epoch = helpers.total_lines_in_all_tfrecord_files(
        dataset_file_list) * train_test_ratio

# Data Parameters
n_hidden_1 = 50 # 1st layer number of nodes
n_hidden_2 = 50 # 2nd layer number of nodes
n_input = 60 # Data input features
n_classes = 2 # Output types. Either laughter or not laughter.

# Construct input pipelines
data, label, test_data, test_label = helpers.input_pipeline(dataset_file_list,
        batch_size, training_epochs, shuffle=False, multithreaded=True,
        train_test_ratio=train_test_ratio)

# Construct model
mlp_train, mlp_test = helpers.multilayer_perceptron(data, test_data, n_input, n_classes, [50, 20])

# Define cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mlp_train,
        labels=label), name="cost_op")
tf.summary.scalar('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
        name='Adam_Optimizer').minimize(cost)

# Initializing local and global variables
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

# Collect all summaries into one handle and create a writer
merged_summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter(save_folder_name)

# Create two handles for accuracy calculation
accuracy, confusion = helpers.accuracy_and_confusion_calculation(mlp_test,
        test_label)

# Collect metadata about the train. Calc times, memory used, device, etc.
run_metadata = tf.RunMetadata()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

# Print time to launch main session
print("Launching TensorFlow Session after " + str(time.time() -
        start_time) + " seconds.")

# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    # Finalize the graph so no more changes can be made
    sess.graph.finalize()

    # Initialize all system variables
    sess.run(init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Write the graph of the system to disk so we can view it later.
    writer.add_graph(sess.graph)

    # Training loop
    batch = 1
    while 1:

        # Try to fetch a new batch. If there are none left, we are done.
        try:

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, s = sess.run([optimizer, cost, merged_summaries],
                    options=run_options, run_metadata=run_metadata)

            if batch % display_step == 0:

                # Display some info about the current training session.
                last_time = cur_time
                cur_time = time.time()
                print("Batch number = {:d}. Epoch ~{:.3f}. {:.3f}s per batch. Total Time = {:.3f}s.".format(batch,
                        batch * batch_size / lines_in_one_epoch, (cur_time -
                        last_time) / display_step, cur_time - start_time))

                # Write data to summary's on each display batch
                writer.add_run_metadata(run_metadata, 'batch' + str(batch))
                writer.add_summary(s, batch)

        except tf.errors.OutOfRangeError:
            break

        batch = batch + 1

    print("Training Completed in " + str(time.time() - start_time) + " seconds.")

    acc, conf = helpers.evaluate_accuracy(sess, accuracy, confusion)
    conf_save_path = os.path.join(save_folder_name, 'pics')
    helpers.save_confusion_matrix(conf, conf_save_path,
            classes=['Not Laughter', 'Laughter'])
    helpers.save_confusion_matrix(conf, conf_save_path,
            classes=['Not Laughter', 'Laughter'], normalize=True,
            name='normalized_confusion_matrix')

    coord.request_stop()
    coord.join(threads)

# Finally, open our TensorBoard tab
helpers.openTensorBoard(save_folder_name)
