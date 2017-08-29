# Other Libs
import os
import webbrowser
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import subprocess
import platform
import re
import itertools
import io
import random
import functools

# My Libs
import plotters

################################################################################
# TensorBoard
################################################################################

def openTensorBoard(path, port=6006):
    """ Opens TensorBoard

    This function kills any processes currently running on the specified port
    and then spawns a TensorBoard subprocess running on that same port. Finally
    it opens a web browser to display the TensorBoard page for the log directory
    'path'.
    """

    # Creates the URL we are interested in making
    urlPath = 'http://localhost:' + str(port)

    # Kills any processes already running on the port we are going to start TensorBoard on.
    _killProcessOnPort(port)

    # Create the TensorBoard process
    proc = subprocess.Popen(['tensorboard', '--logdir=' + path, '--port=' + str(port)])

    # Wait a second
    time.sleep(1)

    # Open the page in our default web browser
    webbrowser.open(urlPath)

def _killProcessOnPort(port):
    """ Kills a process running on the specified port.

    This function kills a process running on the specified port. It should work
    on either Windows or Linux/MacOS systems.
    """

    # Finding the process running on the specified port.
    if "Windows" in platform.system():
        popen = subprocess.Popen(['netstat', '-a','-n','-o'],
                           shell=False,
                           stdout=subprocess.PIPE)
    else:
        popen = subprocess.Popen(['netstat', '-lpn'],
                         shell=False,
                         stdout=subprocess.PIPE)
    (data, err) = popen.communicate()
    data = data.decode("utf-8")

    # Killing the process running on the specified port.
    if "Windows" in platform.system():
        for line in data.split('\n'):
            line = line.strip()
            if '0.0.0.0:' + str(port) in line:
                pid = line.split()[-1]
                subprocess.Popen(['Taskkill', '/PID', pid, '/F'])
    else:
        pattern = "^tcp.*((?:{0})).* (?P<pid>[0-9]*)/.*$"
        pattern = pattern.format(port)
        prog = re.compile(pattern)
        for line in data.split('\n'):
            match = re.match(prog, line)
            if match:
                pid = match.group('pid')
                subprocess.Popen(['kill', '-9', pid])

################################################################################
# Input Pipeline
################################################################################

# Depricated
def read_from_tfrecord(filename_list, num_epochs=None):
    """ Reads data from a number of tfrecords into supported tensors.

    This function is called to create the initial section of the data input
    pipeline. It takes a list of files, and provides a handle to evaluate a
    single record from the dataset, and return it in a useful format.
    """

    # Makes an input string producer for an input pipeline
    filename_queue = tf.train.string_input_producer(filename_list, shuffle=False,
            num_epochs=num_epochs, capacity=100, name='string_input_producer')

    # Make a reader for the queue
    dataset_reader = tf.TFRecordReader(name='dataset_reader')

    # Reads 1000 lines from the text file fetched from the filename_queue
    dataset_key, dataset_value = dataset_reader.read_up_to(filename_queue, num_records=1000, name='read_op')

    test = tf.parse_example(dataset_value, features={
        'data': tf.FixedLenFeature([60], tf.float32),
        'class': tf.FixedLenFeature([1], tf.int64),
        'label': tf.FixedLenFeature([2], tf.int64)
    })

    data = test['data']
    clip = tf.cast(test['class'], tf.int32)
    labels = tf.cast(test['label'], tf.int32)

    return data, clip, labels

# Depricated
def input_pipeline(filename_list, batch_size, num_epochs, shuffle=False,
        multithreaded=True, train_test_ratio=0.85,
        train_test_split_method='file'):

    """ Creates the input pipeline to generate train and test data from records.

    Options:
        filename_list: a list containing the file names of all csv files to be
            used in this training run.
        batch_size: The number of single records in a batch.
        num_epochs: The number of times the whole dataset is to be processed
            before concluding training.
        shuffle: If True, the records for each batch are chosen randomly. If
            False, they are chosen roughly contiguously.
        multithreaded: Chooses whether to run enqueueing operations with
            multiple threads.
        train_test_ratio: The fraction of training data / total data. Defaults
            to 0.85, which is a 85% train 15% test split.
        train_test_split_method:
            'file': This is the default value. Splits data between the test and
                train sets on a per-file basis.

    This input pipeline requires that you have a coordinator to manage the
    queue running threads. Start this at the beginning of training with:

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    And end it at the end of training with:

    # Stop and wait for threads to finish.
    coord.request_stop()
    coord.join(threads)
    """

    # Splits list of filenames into a training and test list with appropriate ratio
    if train_test_split_method == 'file':
        train_filename_list = list(filename_list)
        test_filename_list = []
        while (len(train_filename_list) / len(filename_list)) > train_test_ratio:
            random.shuffle(train_filename_list)
            test_filename_list.append(train_filename_list.pop())
    else:
        print("Not a valid train / test split method.")
        exit(0)

    # Creates the two data outputs from our CSV reader.
    data_point, data_clip, data_point_label = read_from_tfrecord(train_filename_list, num_epochs)
    test_data_point, test_data_clip, test_data_point_label = read_from_tfrecord(test_filename_list, num_epochs)

    # Threads
    if multithreaded == True:
        threads = 4
    else:
        threads = 1

    # Constants
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 10 * batch_size

    # Retrieve two batches. One of data. One of labels. Choose to shuffle or not.
    if (shuffle == False):
        train_feature_batch, train_label_batch = tf.train.batch([data_point,
                data_point_label], batch_size=batch_size, capacity=capacity,
                num_threads=threads, enqueue_many=True,
                name='Training_Batch_Queue')
    else:
        train_feature_batch, train_label_batch = tf.train.shuffle_batch([data_point,
                data_point_label], batch_size=batch_size, capacity=capacity,
                num_threads=threads, min_after_dequeue=min_after_dequeue,
                enqueue_many=True, name='Training_Suffle_Batch_Queue')

    # Batching the test data.
    test_feature_batch, test_label_batch = tf.train.batch([test_data_point,
            test_data_point_label], batch_size=batch_size, capacity=capacity,
            num_threads=threads, enqueue_many=True, name='Test_Batch_Queue')

    return train_feature_batch, train_label_batch, test_feature_batch, test_label_batch

# Depricated
def input_pipeline2(filename_list, batch_size, train_test_split_method='file',
        train_test_ratio=0.85, shuffle=False):

    """ Creates the input pipeline to generate train and test data from records.

    This is the upgraded input_pipeline. It uses tf.contrib.data, and the new
    TensorFlow methods in there to create an input pipeline, rather than the
    old queue method.

    Options:
        filename_list: a list containing the file names of all csv files to be
            used in this training run.
        batch_size: The number of single records in a batch.
        shuffle: If True, the records for each batch are chosen randomly. If
            False, they are chosen roughly contiguously.
        train_test_ratio: The fraction of training data / total data. Defaults
            to 0.85, which is a 85% train 15% test split.
        train_test_split_method:
            'file': This is the default value. Splits data between the test and
                train sets on a per-file basis.
    """

    # Splits list of filenames into a training and test list with appropriate ratio
    if train_test_split_method == 'file':
        train_filename_list = list(filename_list)
        test_filename_list = []
        while (len(train_filename_list) / len(filename_list)) > train_test_ratio:
            random.shuffle(train_filename_list)
            test_filename_list.append(train_filename_list.pop())
    else:
        print("Not a valid train / test split method.")
        exit(0)

    # Define a `tf.contrib.data.Dataset` for iterating over one epoch of the data.
    train_dataset = tf.contrib.data.TFRecordDataset(train_filename_list)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(_parse_function)
    train_dataset = train_dataset.filter(lambda a, b, c, d: tf.equal(tf.shape(a)[0], batch_size))

    test_dataset = tf.contrib.data.TFRecordDataset(test_filename_list)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.map(_parse_function)
    test_dataset = test_dataset.filter(lambda a, b, c, d: tf.equal(tf.shape(a)[0], batch_size))

    # Optionally shuffle
    if shuffle == True:
        train_dataset = train_dataset.shuffle(20000)

    # Return an *initializable* iterator over the dataset, which will allow us to
    # re-initialize it at the beginning of each epoch.
    train_iter = train_dataset.make_initializable_iterator()
    test_iter = test_dataset.make_initializable_iterator()

    return train_iter, test_iter

# Depricated
def _parse_function(proto):
    """ Function mapping a structure of tensors to another structure of tensors.

    I give it the protos from my tfrecord files, and it processes the binary
    into three tensors. One for data, the clip, and the label.
    """

    test = tf.parse_example(proto, features={
        'data': tf.FixedLenFeature([20], tf.float32),
        'class': tf.FixedLenFeature([1], tf.int64),
        'sequence': tf.FixedLenFeature([1], tf.int64),
        'label': tf.FixedLenFeature([2], tf.int64)
    })

    data = test['data']
    clip = tf.cast(test['class'], tf.int32)
    sequence = tf.cast(test['sequence'], tf.int32)
    labels = tf.cast(test['label'], tf.int32)

    return data, clip, sequence, labels

# Depricated
def input_pipeline_data_sequence_creator(data, clip, sequence, label,
        batch_size, window_length, num_features, num_classes,
        run_length='short', shuffle=False):
    """ Takes a batch of data and labels, and creates a batch of sequences.

    This function takes a batch (should be consecutive or the sequences won't
    make any sense) and spits out a series of consecutive sequences made by
    sliding a window over consecutive records of the batch. The sequences are
    'window_length' in length.

    For example, the origional data is sized: [batch_size, num_features] which
    is a 2D matrix of batch_size consecutive records. This function transforms
    this into a 3D matrix of [batch_size, window_length, num_features] where
    batch_size is now: (origional batch_size - window_length + 1).

    EG:
    origional batch:
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15]
    Here, the batch_size = 5, and num_features = 3.

    With a window_length of 3, this becomes:
    [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]
    ],
    [
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12]
    ],
    [
      [7, 8, 9],
      [10, 11, 12],
      [13, 14, 15]
    ]

    As you can see, the size has increased from [5, 3] to [3, 3, 3].

    As for the labels, we simply select the labels of the origional set, but
    chop off (window_length // 2) elements from the start and end of the vector
    to ensure the label length is:
    [origional batch_size - window_length + 1, num_classes].

    Also note, this function takes the variable 'run_length'. This chooses
    between two implimentations of the same process. One takes TensorFlow a
    minute to set up, but halves the time to do this maths. The other is neater
    and runs instantly, however the overhead for dynamically reshaping the
    arrays slows down the code. This is better for short test runs where we want
    to avoid the minute long setup overhead.
    """

    windowed_labels = tf.slice(label, [window_length // 2, 0],
            [batch_size - window_length + 1, num_classes])

    windowed_clip = tf.slice(clip, [window_length // 2, 0],
            [batch_size - window_length + 1, 1])

    windowed_sequence = tf.slice(sequence, [window_length // 2, 0],
            [batch_size - window_length + 1, 1])

    if run_length == 'long':
        list_of_windows_of_data = []
        for x in range(batch_size - window_length + 1):
            list_of_windows_of_data.append(tf.slice(data, [x, 0], [window_length,
                    num_features]))
        windowed_data = tf.squeeze(tf.stack(list_of_windows_of_data, axis=0))
    else:
        windowed_data = tf.map_fn(lambda i: data[i:i + window_length],
                tf.range(batch_size - window_length + 1), dtype=tf.float32)

    return windowed_data, windowed_clip, windowed_sequence, windowed_labels

def input_pipeline3(filename_list, window_length, num_classes, batch_size,
        num_features, train_test_split_method='file', train_test_ratio=0.85,
        shuffle=False, batch_normalize=False):

    """ Takes a list of file names and creates a handle to evaluate input data.

    Input pipeline 3 was creates to supercede input_pipeline2. Input_pipeline2
    used tf.contrib.data.Dataset's to create a chain of commands to create
    batches of input tensors, which were then turned into sequences using
    input_pipeline_data_sequence_creator. This function was slow, and because
    of the seperation between the two files, shuffling couldn't properly be
    done.

    Now, you simply give it a list of file names, and you can pull out a batch
    of variable sized, optionally shuffled and normalization, optionally
    windowed sequences, at high speed. To do so, we use the dataset mapping
    functions _batch_norm, and _sequence_gen, to do the dirty work for us. See
    the comments in those functions for more detailed functionality.
    """

    # Splits list of filenames into a training and test list with appropriate ratio
    if train_test_split_method == 'file':
        train_filename_list = list(filename_list)
        test_filename_list = []
        while (len(train_filename_list) / len(filename_list)) > train_test_ratio:
            random.shuffle(train_filename_list)
            test_filename_list.append(train_filename_list.pop())
    else:
        print("Not a valid train / test split method.")
        exit(0)

    # Define a `tf.contrib.data.Dataset` for iterating over one epoch of the data.
    train_dataset = tf.contrib.data.TFRecordDataset(train_filename_list)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(_parse_function)
    train_dataset = train_dataset.filter(lambda a, b, c, d: tf.equal(tf.shape(a)[0], batch_size))

    test_dataset = tf.contrib.data.TFRecordDataset(test_filename_list)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.map(_parse_function)
    test_dataset = test_dataset.filter(lambda a, b, c, d: tf.equal(tf.shape(a)[0], batch_size))

    # Optionally shuffle
    if shuffle == True:
        train_dataset = train_dataset.shuffle(20000)

    # Optional batch normalization.
    if batch_normalize == True:
        train_dataset = train_dataset.map(_batch_norm)
        test_dataset = test_dataset.map(_batch_norm)

    # Transform datasets into sequences.
    train_dataset = train_dataset.map(functools.partial(_sequence_gen,
            window_length, num_classes, batch_size, num_features))
    test_dataset = test_dataset.map(functools.partial(_sequence_gen,
            window_length, num_classes, batch_size, num_features))

    # Return an *initializable* iterator over the dataset, which will allow us
    # to re-initialize it at the beginning of each epoch.
    train_iter = train_dataset.make_initializable_iterator()
    test_iter = test_dataset.make_initializable_iterator()

    return train_iter, test_iter

def _batch_norm(data, clip, sequence, labels):

    """ Normalizes each batch.

    A function taking 4 different values, and normalizing each MFCC feature
    individually. Returns other non-data handles unchanged.
    """

    mean, variance = tf.nn.moments(data, [0])
    normalized_data = tf.nn.batch_normalization(data, mean, variance, None,
            None, 1e-3)

    return (normalized_data, clip, sequence, labels)

def _sequence_gen(window_length, num_classes, batch_size, num_features, data,
        clip, sequence, labels):

    """ Takes a batch of data and labels, and creates a batch of sequences.

    This function performs the work of input_pipeline_data_sequence_creator, but
    in a nice and easy function which can be called by the tf.contrib.data
    Dataset map function.

    This function takes a batch (should be consecutive or the sequences won't
    make any sense) and spits out a series of consecutive sequences made by
    sliding a window over consecutive records of the batch. The sequences are
    'window_length' in length.

    For example, the origional data is sized: [batch_size, num_features] which
    is a 2D matrix of batch_size consecutive records. This function transforms
    this into a 3D matrix of [batch_size, window_length, num_features] where
    batch_size is now: (origional batch_size - window_length + 1).

    EG:
    origional batch:
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15]
    Here, the batch_size = 5, and num_features = 3.

    With a window_length of 3, this becomes:
    [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]
    ],
    [
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12]
    ],
    [
      [7, 8, 9],
      [10, 11, 12],
      [13, 14, 15]
    ]

    As you can see, the size has increased from [5, 3] to [3, 3, 3].

    As for the labels, we simply select the labels of the origional set, but
    chop off (window_length // 2) elements from the start and end of the vector
    to ensure the label length is:
    [origional batch_size - window_length + 1, num_classes].

    Also note, this function takes the variable 'run_length'. This chooses
    between two implimentations of the same process. One takes TensorFlow a
    minute to set up, but halves the time to do this maths. The other is neater
    and runs instantly, however the overhead for dynamically reshaping the
    arrays slows down the code. This is better for short test runs where we want
    to avoid the minute long setup overhead.
    """

    windowed_labels = tf.slice(labels, [window_length // 2, 0],
            [batch_size - window_length + 1, num_classes])

    windowed_clip = tf.slice(clip, [window_length // 2, 0],
            [batch_size - window_length + 1, 1])

    windowed_sequence = tf.slice(sequence, [window_length // 2, 0],
            [batch_size - window_length + 1, 1])

    list_of_windows_of_data = []
    for x in range(batch_size - window_length + 1):
        list_of_windows_of_data.append(tf.slice(data, [x, 0], [window_length,
                num_features]))
    windowed_data = tf.squeeze(tf.stack(list_of_windows_of_data, axis=0))

    return (windowed_data, windowed_clip, windowed_sequence, windowed_labels)

################################################################################
# Models
################################################################################

def multilayer_perceptron(x_train, x_test, n_inputs, n_outputs, hidden_layers,
        activation_function='relu', output_layer_biases=True):
    """ Creates a multi layer perceptron for training.

    This function creates a MLP for training all in one function. It creates
    nodes with weights and biases for each previous layer, to create a fully
    connected MLP. It creates two seperate data flows, one for training data,
    and another for test data to evaluate the test set. It creates summarys for
    all node weights and biases.

    Abstracting the MLP creation allows us to create a number of different sized
    MLP's and test them individually and quickly without worrying too much about
    coding when changing them.

    Having two data flows through the same MLP might seem complicated, but it
    improves the efficiency of the system by 15 percent by letting us pass data
    in straight from tensorflow and not feed it into a session in a feed_dict.
    It also allows us to not modify the graph structure mid-session, or need to
    start additional sessions.

    Arguments:
        x_train: This is a handle for a batch of training data.
        x_test: This is a handle for a batch of test data.
        n_inputs: This is an integer. It is the number of inputs in the dataset.
        n_outputs: This is an integer. It is the number of outputs of the MLP.
        hidden_layers: This is a list of integers. It is the number of nodes
            in each hidden layer of the MLP.
        activation_function: This is 'relu' by default but 'sigmoid' is also
            supported. This is to choose the activation function between layers.

    Returns:
        out_layer_train: This is a handle for the output of the MLP from inputs
            into the x_train input.
        out_layer_test: This is a handle for the output of the MLP from inputs
            into the

    """

    # Variable initialization values
    w_mean = 0.0
    w_std = 1.0
    b_mean = 0.0
    b_std = 1.0

    # Store layers weight & biases in dictionaries
    weights = {}
    biases = {}

    # Select the appropriate activation function
    if activation_function == 'sigmoid':
        a = lambda x: tf.sigmoid(x)
    elif activation_function == 'relu':
        a = lambda x: tf.nn.relu(x)
    elif activation_function == 'tanh':
        a = lambda x: tf.tanh(x)
    else:
        a = lambda x: tf.nn.relu(x)

    if clipped == True:
        b = lambda x: tf.minimum(a(x), lim)
    else:
        b = a

    if dropout == True:
        c = lambda x: tf.nn.dropout(b(x), keep_prob)
    else:
        c = b

    op = c

    # Add the first layer to the dictionary
    cur_layer_num = 1
    with tf.variable_scope("Layer1"):
        weights['w' + str(cur_layer_num)] = tf.Variable(tf.random_normal(
                [n_inputs, hidden_layers[0]], mean=w_mean, stddev=w_std),
                name=("Layer_" + str(cur_layer_num) + "_Weights"))
        biases['b' + str(cur_layer_num)] = tf.Variable(tf.random_normal(
                [hidden_layers[0]], mean=b_mean, stddev=b_std), name=("Layer_" +
                str(cur_layer_num) + "_Biases"))

    # Add all but the last layers to the dictionary
    cur_layer_num = 2
    for layer in hidden_layers[1:]:
        with tf.variable_scope("Layer" + str(cur_layer_num)):
            weights['w' + str(cur_layer_num)] = tf.Variable(tf.random_normal(
                    [hidden_layers[cur_layer_num - 2], layer], mean=w_mean,
                    stddev=w_std), name=("Layer_" + str(cur_layer_num) +
                    "_Weights"))
            biases['b' + str(cur_layer_num)] = tf.Variable(tf.random_normal(
                    [layer], mean=b_mean, stddev=b_std), name=("Layer_" +
                    str(cur_layer_num) + "_Biases"))
            cur_layer_num += 1

    # Add the last layer to the dictionary
    with tf.variable_scope("OutputLayer"):
        weights['out'] = tf.Variable(tf.random_normal([hidden_layers[-1],
                n_outputs], mean=w_mean, stddev=w_std), name="Output_Weights")
        biases['out'] = tf.Variable(tf.random_normal([n_outputs], mean=b_mean,
                stddev=b_std), name="Output_Biases")

    # Set up the connections for the first layer
    cur_layer_num = 1
    cur_layer_train = tf.add(tf.matmul(x_train, weights['w1']), biases['b1'])
    cur_layer_train = op(cur_layer_train)
    cur_layer_test = tf.add(tf.matmul(x_test, weights['w1']), biases['b1'])
    cur_layer_test = op(cur_layer_test)
    with tf.variable_scope("Layer_1_Summarys"):
        tf.summary.image("Weights", tf.reshape(weights['w1'], [1, n_inputs, hidden_layers[0], 1]))
        tf.summary.image("Biases", tf.reshape(biases['b1'], [1, 1, hidden_layers[0], 1]))
        tf.summary.histogram("Weights", weights['w1'])
        tf.summary.histogram("Biases", biases['b1'])

    # Set up the connections for the middle layers
    cur_layer_num = 2
    for layer in hidden_layers[1:]:
        cur_layer_train = tf.add(tf.matmul(cur_layer_train, weights['w' +
                str(cur_layer_num)]), biases['b' + str(cur_layer_num)])
        cur_layer_train = op(cur_layer_train)
        cur_layer_test = tf.add(tf.matmul(cur_layer_test, weights['w' +
                str(cur_layer_num)]), biases['b' + str(cur_layer_num)])
        cur_layer_test = op(cur_layer_test)
        with tf.variable_scope("Layer_" + str(cur_layer_num) + "_Summarys"):
            tf.summary.image("Weights", tf.reshape(weights['w' +
                    str(cur_layer_num)], [1, hidden_layers[cur_layer_num - 2],
                    layer, 1]))
            tf.summary.image("Biases", tf.reshape(biases['b' +
                    str(cur_layer_num)], [1, 1, layer, 1]))
            tf.summary.histogram("Weights", weights['w' + str(cur_layer_num)])
            tf.summary.histogram("Biases", biases['b' + str(cur_layer_num)])
        cur_layer_num += 1

    # Set up the connections for the last layer
    if output_layer_biases == True:
        cur_layer_train = tf.add(tf.matmul(cur_layer_train, weights['out']), biases['out'])
        cur_layer_test = tf.add(tf.matmul(cur_layer_test, weights['out']), biases['out'])
    else:
        cur_layer_train = tf.matmul(cur_layer_train, weights['out'])
        cur_layer_test = tf.matmul(cur_layer_test, weights['out'])

    with tf.variable_scope("Output_Summarys"):
        tf.summary.image("Weights", tf.reshape(weights['out'], [1, hidden_layers[-1], n_outputs, 1]))
        tf.summary.histogram("Weights", weights['out'])
        if output_layer_biases == True:
            tf.summary.image("Biases", tf.reshape(biases['out'], [1, 1, n_outputs, 1]))
            tf.summary.histogram("Biases", biases['out'])

    return cur_layer_train, cur_layer_test

def sequence_mlp(x_train, x_test, n_features, window_length, n_outputs,
        batch_size, hidden_layers, activation_function='relu',
        output_layer_biases=True, initialization=None, clipped=False,
        dropout=False):
    """ Creates an MLP for multiple data frames at once.

    The goal of this function is to mimic the multilayer_perceptron function in
    this file in terms of functionality and usage, but it needs some small
    modifications to the first layer, to attach it correctly the modified input
    pipeline for sequential data.

    This function creates a MLP for training all in one function. It creates
    nodes with weights and biases for each previous layer, to create a fully
    connected MLP. It creates two seperate data flows, one for training data,
    and another for test data to evaluate the test set. It creates summarys for
    all node weights and biases.

    Arguments:
        x_train: This is a handle for a batch of training data.
        x_test: This is a handle for a batch of test data.
        n_features: This is an integer. It is the number of features in each
            record.
        window_length: This is the number of records in a sequence.
        n_outputs: This is an integer. It is the number of outputs of the MLP.
        hidden_layers: This is a list of integers. It is the number of nodes
            in each hidden layer of the MLP.
        activation_function: This is 'relu' by default but 'sigmoid' is also
            supported. This is to choose the activation function between layers.

    Returns:
        out_layer_train: This is a handle for the output of the MLP from inputs
            into the x_train input.
        out_layer_test: This is a handle for the output of the MLP from inputs
            into the
    """

    # The default initialization parameters for the layers
    w_mean = 0.0
    w_std = 0.1
    b_mean = 0.0
    b_std = 0.0

    # Clipping max
    lim = 10
    keep_prob = 0.2

    # Store layers weight & biases in dictionaries
    weights = {}
    biases = {}

    # Select the appropriate activation function
    if activation_function == 'sigmoid':
        a = lambda x: tf.sigmoid(x)
    elif activation_function == 'relu':
        a = lambda x: tf.nn.relu(x)
    elif activation_function == 'tanh':
        a = lambda x: tf.tanh(x)
    else:
        a = lambda x: tf.nn.relu(x)

    if clipped == True:
        b = lambda x: tf.minimum(a(x), lim)
    else:
        b = a

    if dropout == True:
        c = lambda x: tf.nn.dropout(b(x), keep_prob)
    else:
        c = b

    op = c

    # Add the first layer to the dictionary
    cur_layer_num = 1
    with tf.variable_scope("Layer1"):
        if initialization == 'Xavier':
            w_std = tf.sqrt(3 / (n_features + hidden_layers[0]))
            b_std = 0.0
            w_mean = 0.0
            b_mean = 0.0
        weights['w' + str(cur_layer_num)] = tf.Variable(tf.random_normal([n_features * window_length,
                hidden_layers[0]], mean=w_mean, stddev=w_std), name=("Layer_" +
                str(cur_layer_num) + "_Weights"))
        biases['b' + str(cur_layer_num)] = tf.Variable(tf.random_normal(
                [hidden_layers[0]], mean=b_mean, stddev=b_std), name=("Layer_" +
                str(cur_layer_num) + "_Biases"))

    # Add all but the last layers to the dictionary
    cur_layer_num = 2
    for layer in hidden_layers[1:]:
        with tf.variable_scope("Layer" + str(cur_layer_num)):
            if initialization == 'Xavier':
                w_std = tf.sqrt(3 / (layer + hidden_layers[cur_layer_num - 2]))
                b_std = 0.0
                w_mean = 0.0
                b_mean = 0.0
            weights['w' + str(cur_layer_num)] = tf.Variable(tf.random_normal(
                    [hidden_layers[cur_layer_num - 2], layer], mean=w_mean,
                    stddev=w_std), name=("Layer_" + str(cur_layer_num) +
                    "_Weights"))
            biases['b' + str(cur_layer_num)] = tf.Variable(tf.random_normal(
                    [layer], mean=b_mean, stddev=b_std), name=("Layer_" +
                    str(cur_layer_num) + "_Biases"))
            cur_layer_num += 1

    # Add the last layer to the dictionary
    with tf.variable_scope("OutputLayer"):
        if initialization == 'Xavier':
            w_std = tf.sqrt(3 / (hidden_layers[-1] + n_outputs))
            b_std = 0.0
            w_mean = 0.0
            b_mean = 0.0
        weights['out'] = tf.Variable(tf.random_normal([hidden_layers[-1],
                n_outputs], mean=w_mean, stddev=w_std), name="Output_Weights")
        biases['out'] = tf.Variable(tf.random_normal([n_outputs], mean=b_mean,
                stddev=b_std), name="Output_Biases")

    # Set up the connections for the first layer
    cur_layer_num = 1
    cur_layer_train = tf.add(tf.matmul(tf.reshape(x_train, [batch_size -
            window_length + 1, n_features * window_length]), weights['w1']),
            biases['b1'])
    cur_layer_train = op(cur_layer_train)
    cur_layer_test = tf.add(tf.matmul(tf.reshape(x_test, [batch_size -
            window_length + 1, n_features * window_length]), weights['w1']),
            biases['b1'])
    cur_layer_test = op(cur_layer_test)
    with tf.variable_scope("Layer_1_Summarys"):
        image_tensor = tf.reshape(weights['w1'], [1, n_features * window_length, hidden_layers[0], 1])
        tf.summary.image("Weights", image_tensor)
        tf.summary.image("Biases", tf.reshape(biases['b1'], [1, 1, hidden_layers[0], 1]))
        tf.summary.image("Reshaped Weights", tf.reshape(tf.reduce_mean(image_tensor, 2), [1, window_length, n_features, 1]))
        tf.summary.histogram("Weights", weights['w1'])
        tf.summary.histogram("Biases", biases['b1'])

    # Set up the connections for the middle layers
    cur_layer_num = 2
    for layer in hidden_layers[1:]:
        cur_layer_train = tf.add(tf.matmul(cur_layer_train, weights['w' +
                str(cur_layer_num)]), biases['b' + str(cur_layer_num)])
        cur_layer_train = op(cur_layer_train)
        cur_layer_test = tf.add(tf.matmul(cur_layer_test, weights['w' +
                str(cur_layer_num)]), biases['b' + str(cur_layer_num)])
        cur_layer_test = op(cur_layer_test)
        with tf.variable_scope("Layer_" + str(cur_layer_num) + "_Summarys"):
            image_tensor = tf.reshape(weights['w' + str(cur_layer_num)],
                    [1, hidden_layers[cur_layer_num - 2], layer, 1])
            tf.summary.image("Weights", image_tensor)
            tf.summary.image("Biases", tf.reshape(biases['b' +
                    str(cur_layer_num)], [1, 1, layer, 1]))
            tf.summary.image("Reshaped Weights", tf.reshape(tf.reduce_mean(image_tensor, 2), [1, window_length, n_features, 1]))
            tf.summary.histogram("Weights", weights['w' + str(cur_layer_num)])
            tf.summary.histogram("Biases", biases['b' + str(cur_layer_num)])
        cur_layer_num += 1

    # Set up the connections for the last layer
    if output_layer_biases == True:
        cur_layer_train = tf.add(tf.matmul(cur_layer_train, weights['out']), biases['out'])
        cur_layer_test = tf.add(tf.matmul(cur_layer_test, weights['out']), biases['out'])
    else:
        cur_layer_train = tf.matmul(cur_layer_train, weights['out'])
        cur_layer_test = tf.matmul(cur_layer_test, weights['out'])

    with tf.variable_scope("Output_Summarys"):
        image_tensor = tf.reshape(weights['out'], [1, hidden_layers[-1], n_outputs, 1])
        tf.summary.image("Weights", image_tensor)
        tf.summary.histogram("Weights", weights['out'])
        if output_layer_biases == True:
            tf.summary.image("Biases", tf.reshape(biases['out'], [1, 1, n_outputs, 1]))
            tf.summary.histogram("Biases", biases['out'])

    return cur_layer_train, cur_layer_test

def ltsm_model(data_input, num_features, num_classes, n_hidden=128):
    """ Creates an LTSM model to classify a sequence of audio data.

    Inputs:
        data_input: The input data handle. The format for this is shown below.
        n_hidden: THe number of hidden units in the LTSM state.
    """

    # Initialisation Parameters
    w_mean = 0.0
    w_std = 2.0
    b_mean = 0.0
    b_std = 0.5

    # Variables
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, num_classes], mean=w_mean, stddev=w_std))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes], mean=b_mean, stddev=b_std))
    }

    # RNN
    cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
    val, state = tf.nn.dynamic_rnn(cell, data_input, dtype=tf.float32)

    # Getting only the last output
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    data_prediction = tf.matmul(last, weights['out']) + biases['out']

    return data_prediction

def cost_function(label, prediction, cost_type='entropy'):

    weighted_labels = tf.multiply(label, tf.constant([1, 30], dtype=tf.int32), name='add_weight_to_labels')

    if (cost_type == 'entropy'):
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction, labels=weighted_labels, name="cost_op"))
    elif (cost_type == 'mean_squared'):
        cost = tf.reduce_mean(tf.squared_difference(prediction, tf.cast(weighted_labels, tf.float32)))

    tf.summary.scalar('cost', cost)

    return cost

################################################################################
# Metrics
################################################################################

def accuracy_and_confusion_calculation(prediction, label):
    """ Adds calculation of accuracy and confusion to the tensorflow graph.

    These two functions are in the same function because they can re-use graph
    nodes. Could maybe be seperated in the future.
    """

    prediction_argmax = tf.argmax(prediction, 1, name='prediction_argmax')
    label_argmax = tf.argmax(label, 1, name='label_argmax')

    # Test model
    correct_prediction = tf.equal(prediction_argmax, label_argmax, name='equal')

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float",
            name='accuracy_cast_to_float'), name='accuracy_calculate_mean')

    confusion = tf.confusion_matrix(prediction_argmax, label_argmax,
            name='confusion_matrix', num_classes=2)

    return accuracy, confusion

def evaluate_accuracy(sess, accuracy, confusion):
    """ This function evaluates the accuracy of the model on the whole test set.

    Could be improved much further by potentially using the
    tf.contrib.streaming_accuracy function, and/or while loops inside tf, and
    by attaching a scalar summary to one of these tf functions.
    """

    acc_list = []
    conf_list = []
    batch_count = 1
    try:
        while 1:

            # Evaluate accuracy and confusion
            acc, conf = sess.run([accuracy, confusion])

            # Append accuracy and confidence to our lists
            acc_list.append(acc)
            conf_list.append(conf)

            if (batch_count % 10 == 0):
                print("Evaluating Accuracy for Test Batch " + str(batch_count))

            batch_count += 1

    except:

        # Print the final average of acc_list
        final_accuracy = sum(acc_list) / len(acc_list)
        print("Accuracy: " + str(final_accuracy))

        # Print the final confusion of conf_list
        final_conf = add_2d_tensor_list(conf_list)

    return final_accuracy, final_conf

def streaming_accuracy_and_confusion_calculation(label, prediction,
        num_classes):

    """ A function which defines streaming accuracy and streaming confusion.

    Call test_op for all batches in one epoch, then evaluate the confusion or
    the accuracy to get their values. This calculates the accuracy and confusion
    while keeping all calculation in the TensorFlow backend.
    """

    # Get the argmax of the labels and predictions.
    prediction_argmax = tf.argmax(prediction, 1, name='prediction_argmax')
    label_argmax = tf.argmax(label, 1, name='label_argmax')

    with tf.name_scope("test_batch_ops"):

        # the streaming accuracy (lookup and update tensors)
        accuracy, accuracy_update = tf.metrics.accuracy(label_argmax,
                prediction_argmax, name='accuracy')

        # Compute a per-batch confusion
        batch_confusion = tf.confusion_matrix(label_argmax, prediction_argmax,
                num_classes=num_classes, name='batch_confusion')

        # Create an accumulator variable to hold the counts
        confusion = tf.Variable(tf.zeros([num_classes, num_classes],
                dtype=tf.int32), name='confusion')

        reset_conf_op = tf.assign(confusion, tf.zeros([num_classes,
                num_classes], name='reset_conf_zeros', dtype=tf.int32),
                name='reset_conf_assign')

        # Create the update op for doing a "+=" accumulation on the batch
        confusion_update = confusion.assign(confusion + batch_confusion)

        # Combine streaming accuracy and confusion matrix updates in one op
        test_op = tf.group(accuracy_update, confusion_update)

    stream_vars = [i for i in tf.local_variables() if i.name.split('/')[0] == 'test_batch_ops']
    reset_acc_op = tf.variables_initializer(stream_vars)
    reset_op = tf.group(reset_acc_op, reset_conf_op)

    return test_op, reset_op, accuracy, confusion

################################################################################
# Misc
################################################################################

def load_data_into_ram():
    """ A function that loads all data into numpy arrays in RAM.

    Useful for only really small datasets.
    """

    data_path = os.path.join(directory, '../../Dataset/laughter_data_small.csv')
    labels_path = os.path.join(directory, '../../Dataset/laughter_labels_nn_small.csv')

    data = np.loadtxt(data_path, delimiter=',', dtype=float)
    data_labels = np.loadtxt(labels_path, delimiter=',', dtype=float)

def total_lines_in_all_csv_files(file_list):
    """Calculates the total number of lines is all the files.

    It expects a list of strings of all the input file names to scan though.
    """

    count = 0
    for file in file_list:
        with open(file) as f:
            for i, l in enumerate(f):
                pass
        count += (i + 1)
    return count

def total_lines_in_all_tfrecord_files(file_list, recalculate=False):
    """Calculates the total number of lines is all the files.

    It expects a list of strings of all the input file names to scan though.
    If recalculate is left as false, the number of lines is simply read from a
    file. If that file doesn't exist, then it is recalculated. If this flag is
    changed to true, then it is recalculated no matter what.
    """

    if recalculate == False:
        path = os.path.join(os.path.dirname(file_list[0]), 'tot_records.txt')
        if (os.path.isfile(path)):
            f = open(path, 'r')
            length = int(f.read())
            f.close()

            return length

    count = 0
    file_num = 0
    for filename in file_list:

        file_num += 1

        print("Checking File: " + str(file_num) + "/" + str(len(file_list)))

        # Open a reader
        reader = tf.python_io.tf_record_iterator(filename)

        # Count number of lines in it
        for line in reader:
            count += 1

    f = open(path, 'w')
    f.write(str(count))
    f.close()

    return count

def read_from_csv(filename_list, num_epochs=None):
    """ Reads data from a number of csv files into supported tensors.

    This function is called to create the initial section of the data input
    pipeline. It takes a list of files, and provides a handle to evaluate a
    single record from the dataset, and return it in a useful format.
    """

    # Makes an input string producer for an input pipeline
    filename_queue = tf.train.string_input_producer(filename_list, shuffle=False,
            num_epochs=num_epochs, capacity=10000, name='string_input_producer')

    # Make a reader for the queue
    dataset_reader = tf.TextLineReader(name='dataset_reader')
    # Reads a single random line from the text file fetched from the filename_queue
    dataset_key, dataset_value = dataset_reader.read(filename_queue, name='single_read_op')

    # The default values for the records
    dataset_record_defaults = [[1.0] for x in range (60)]
    dataset_record_defaults.append([0])
    dataset_record_defaults.append([1])
    dataset_record_defaults.append([0]) # Defaults to not laughter

    # Decode the single line
    data_list = tf.decode_csv(dataset_value, record_defaults=dataset_record_defaults, name='csv_data_decoder')

    # Stack the columns together into tensors
    data = tf.stack(data_list[0:60:1], name='data_stack')
    clip = tf.stack(data_list[60], name='clip_stack')
    labels = tf.stack(data_list[61:63:1], name='label_stack')

    return data, clip, labels

def get_save_dir(directory, save_folder_name):
    """ A function which returns the name of an empty folder to store metadata.

    TensorBoard doesn't like it when there exists an events file already in the
    folder being used to store TensorBoard data, and overrides it. To store
    data from many runs, you need to create multiple folders. This function
    takes a directory, and the name of the save folder, and basically appends
    a number to the end of the name if that folder exists, and keeps increasing
    that number until a unique name can be found.
    """

    dirList = os.listdir(directory)
    dirName = save_folder_name
    maxNum = 0
    while 1:
        flag = 0
        for d in dirList:
            if dirName + str(maxNum) in d:
                maxNum += 1
                flag = 1
        if (flag == 0):
            break
    name = os.path.join(directory, dirName + str(maxNum))
    return name

def add_2d_tensor_list(tensor_list):
    """ Takes a list of 2D tensors. Does elementwise addition with them.

    """

    # Get size of matrix
    y, x = np.shape(tensor_list[0])

    # Create resultant matrix
    result = np.zeros([y, x], dtype=int)

    # Sum all elements of tensors in tensor_list and put in result.
    for tensor in tensor_list:
        for a in range(y):
            for b in range(x):
                try:
                    result[b][a] += tensor[b][a]
                except:
                    print("Something went wrong when trying to create the confusion matrix.")
                    print("One tensor is the wrong size")
                    print(tensor_list)
                    print("$$$$$$$$$$$$$$$$$")
                    print(tensor)

    return result

def print_system_params(argv):

    param_dict_keylist = [
            "name",
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "training_epochs",
            "display_step",
            "batch_size",
            "train_test_ratio",
            "activation_function",
            "layers",
            "output_layer_biases",
            "n_input",
            "n_classes"
    ]

    for x in range(14):
        print(param_dict_keylist[x] + ": " + argv[x + 1])

def numpy_long_output():
    np.set_printoptions(threshold=np.nan)

################################################################################
# Classes
################################################################################

class Metrics(object):
    """ An object containing metrics through a number of epochs.

    """

    def __init__(self):
        self.sensitivity_and_specificity_list = []

    def end_of_epoch_sens_spec(self, sess, accuracy, confusion, test_op,
            reset_op, logger, cur_epoch_num, pics_save_path):
        """ Performs all end of epoch calculations.

        This function was made to perform all calculations for the sensitivity
        and specificity all at once. This saves a bunch of space in the actual
        network file and makes it all look much neater.
        """

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
                sens, spec = self.sensitivity_and_specificity(conf)
                print("Sensitivity: {:.3f}, Specificity: {:.3f}".format(sens, spec))
                logger.log_scalar("Sensitivity", sens, cur_epoch_num)
                logger.log_scalar("Specificity", spec, cur_epoch_num)

                # Reset streaming metrics
                sess.run(reset_op)
                break

        return conf

    def sensitivity_and_specificity(self, conf):
        """ Calculates the sensitivity and specificity given a confusion matrix.

        Takes a 2x2 matrix, where the columns represent predictions, and the rows
        represent the true result. So the matrix looks like:

        TN, FP
        FN, TP

        Where TN is true negative, and FP is false positive.

        Sensitivity is TP/(TP + FN) and is the true positive rate.
        Specificity is TN/(TN + FP) and is the true negative rate.
        """

        try:
            assert np.shape(conf) == (2, 2)
        except:
            return None, None

        if ((conf[1][1] + conf[1][0]) == 0):
            sens = 0
        else:
            sens = conf[1][1] / (conf[1][1] + conf[1][0])

        if ((conf[0][0] + conf[0][1]) == 0):
            spec = 0
        else:
            spec = conf[0][0] / (conf[0][0] + conf[0][1])

        self.sensitivity_and_specificity_list.append((sens, spec))

        return sens, spec

    def get_max_sensitivity_and_specificity(self):
        """ Returns the maximum sensitivity and specificity for a whole run.

        Returns the maximum sum of the sensitivity and specificity for a whole
        training run.
        """

        maximum = (0.0, 0.0, 0)
        epoch = 0
        for sens, spec in self.sensitivity_and_specificity_list:
            if (sens + spec) > (maximum[0] + maximum[1]):
                maximum = (sens, spec, epoch)
            epoch += 1

        return maximum

class Logger(object):
    """Logging in tensorboard without tensorflow ops.

    A class by Michael Gygli to log things which are not tensorflow ops. EG:
    just random numbers, or other things not related to the tensorflow graph.
    """

    def __init__(self, log_dir, writer=None):
        """Creates a summary writer logging to log_dir."""
        if writer == None:
            self.writer = tf.summary.FileWriter(log_dir)
        else:
            self.writer = writer

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_images(self, tag, images, step):
        """Logs a list of images."""

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = io.StringIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)


    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
