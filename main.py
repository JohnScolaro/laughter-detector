import tensorflow as tf
import numpy as np
import matplotlib as mpl
import os
import helpers
import time

start_time = time.time()
directory = os.path.dirname(__file__)
save_folder_name = helpers.get_save_dir(os.path.join(directory, 'tensorboard'), "mlp_test")

# Make lists of file names
dataset_file_list = []
for x in range(100): #100
    name = os.path.join(directory, '../../Dataset/') + str(x) + "_dataset.tfrecord"
    if os.path.isfile(name):
        dataset_file_list.append(name)

# Start up a logger to log misc variables.
logger = helpers.Logger(save_folder_name)

# System Parameters
learning_rate = 0.1
training_epochs = 2
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

x = tf.placeholder("float", [None, n_input], name='model_input_placeholder')
y = tf.placeholder("float", [None, n_classes], name='our_label_placeholder')

def multilayer_perceptron(x_train, x_test, weights, biases):

    # Hidden layer with RELU activation
    layer_1_train = tf.add(tf.matmul(x_train, weights['h1']), biases['b1'])
    layer_1_test = tf.add(tf.matmul(x_test, weights['h1']), biases['b1'])
    layer_1_train = tf.nn.relu(layer_1_train)
    layer_1_test = tf.nn.relu(layer_1_test)
    tf.summary.image("Layer 1 Weights", tf.reshape(weights['h1'], [1, n_input, n_hidden_1, 1]))
    tf.summary.image("Layer 1 Biases", tf.reshape(biases['b1'], [1, 1, n_hidden_1, 1]))
    tf.summary.histogram("Layer 1 Weights", weights['h1'])
    tf.summary.histogram("Layer 1 Biases", biases['b1'])

    # Hidden layer with RELU activation
    layer_2_train = tf.add(tf.matmul(layer_1_train, weights['h2']), biases['b2'])
    layer_2_train = tf.nn.relu(layer_2_train)
    layer_2_test = tf.add(tf.matmul(layer_1_test, weights['h2']), biases['b2'])
    layer_2_test = tf.nn.relu(layer_2_test)
    tf.summary.image("Layer 2 Weights", tf.reshape(weights['h2'], [1, n_hidden_1, n_hidden_2, 1]))
    tf.summary.image("Layer 2 Biases", tf.reshape(biases['b2'], [1, 1, n_hidden_2, 1]))
    tf.summary.histogram("Layer 2 Weights", weights['h2'])
    tf.summary.histogram("Layer 2 Biases", biases['b2'])

    # Output layer with linear activation
    out_layer_train = tf.matmul(layer_2_train, weights['out']) + biases['out']
    out_layer_test = tf.matmul(layer_2_test, weights['out']) + biases['out']
    tf.summary.image("Output Weights", tf.reshape(weights['out'], [1, n_hidden_2, n_classes, 1]))
    tf.summary.image("Output Biases", tf.reshape(biases['out'], [1, 1, n_classes, 1]))
    tf.summary.histogram("Output Weights", weights['out'])
    tf.summary.histogram("Output Biases", biases['out'])

    return out_layer_train, out_layer_test

# Store layers weight & biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="Layer_1_Weights"),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="Layer_2_Weights"),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="Output_Weights")
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="Layer_1_Biases"),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name="Layer_2_Biases"),
    'out': tf.Variable(tf.random_normal([n_classes]), name="Output_Biases")
}

# Construct input pipeline
data, label, test_data, test_label = helpers.input_pipeline(dataset_file_list,
        batch_size, training_epochs, shuffle=False, multithreaded=True,
        train_test_ratio=train_test_ratio)

# Construct model
mlp_train, mlp_test = multilayer_perceptron(data, test_data, weights, biases)

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
accuracy, confusion = helpers.accuracy_calculation(mlp_test, y)

# Collect metadata about the train. Calc times, memory used, device, etc.
run_metadata = tf.RunMetadata()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

# Print time to launch main session
print("Launching TensorFlow Session after " + str(time.time() -
        start_time) + " seconds.")

# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

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

            #data_batch, label_batch = sess.run([data, label])
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, s = sess.run([optimizer, cost, merged_summaries],
                    #feed_dict = {x: data_batch, y: label_batch},
                    options=run_options, run_metadata=run_metadata)

            if batch % display_step == 0:

                # Display logs per epoch every display step
                print("Batch number = {:d}. Epoch ~{:.3f}.".format(batch, batch *
                        batch_size / lines_in_one_epoch))

                # Write data to summary's on each display step
                writer.add_run_metadata(run_metadata, 'batch' + str(batch))
                writer.add_summary(s, batch)

        except tf.errors.OutOfRangeError:
            break

        batch = batch + 1

    print("Training Completed in " + str(time.time() - start_time) + " seconds.") # 44.86 seconds with a feed_dict, 40.27 without, and 38 in the funky configuration

        # acc, conf = helpers.evaluate_accuracy(sess, accuracy, confusion,
        #         test_data, test_label)
        # conf_save_path = os.path.join(save_folder_name, 'pics')
        # helpers.save_confusion_matrix(conf, conf_save_path,
        #         classes=['Not Laughter', 'Laughter'])
        # helpers.save_confusion_matrix(conf, conf_save_path,
        #         classes=['Not Laughter', 'Laughter'], normalize=True,
        #         name='normalized_confusion_matrix')

    coord.request_stop()
    coord.join(threads)

# Finally, open our TensorBoard tab
helpers.openTensorBoard(save_folder_name)
