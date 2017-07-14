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

def openTensorBoard(path, port = 6006):
    """ Opens TensorBoard

    This function kills any processes currently running on the specified port
    and then spawns a TensorBoard subprocess running on that same port. Finally
    it opens a web browser to display the TensorBoard page for the log directory
    'path'.
    """

    # Creates the URL we are interested in making
    urlPath = 'http://localhost:' + str(port)

    # Kills any processes already running on the port we are going to start TensorBoard on.
    killProcessOnPort(port)

    # Create the TensorBoard process
    proc = subprocess.Popen(['tensorboard', '--logdir=' + path, '--port=' + str(port)])

    # Wait a second
    time.sleep(1)

    # Open the page in our default web browser
    webbrowser.open(urlPath)

def killProcessOnPort(port):
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

    test_dataset = tf.contrib.data.TFRecordDataset(test_filename_list)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.map(_parse_function)

    # Optionally shuffle
    if shuffle == True:
        train_dataset = train_dataset.shuffle(20000)
        test_dataset = test_dataset.shuffle(20000)

    # Return an *initializable* iterator over the dataset, which will allow us to
    # re-initialize it at the beginning of each epoch.
    train_iter = train_dataset.make_initializable_iterator()
    test_iter = test_dataset.make_initializable_iterator()

    return train_iter, test_iter

def _parse_function(proto):
    """ Function mapping a structure of tensors to another structure of tensors.

    I give it the protos from my tfrecord files, and it processes the binary
    into three tensors. One for data, the clip, and the label.
    """

    test = tf.parse_example(proto, features={
        'data': tf.FixedLenFeature([60], tf.float32),
        'class': tf.FixedLenFeature([1], tf.int64),
        'label': tf.FixedLenFeature([2], tf.int64)
    })

    data = test['data']
    clip = tf.cast(test['class'], tf.int32)
    labels = tf.cast(test['label'], tf.int32)

    return data, clip, labels

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

    # Store layers weight & biases in dictionaries
    weights = {}
    biases = {}
    sig = lambda x: tf.sigmoid(x)
    relu = lambda x: tf.nn.relu(x)

    # Select the appropriate activation function
    if activation_function == 'relu':
        op = relu
    elif activation_function == 'sigmoid':
        op = sig
    else:
        op = relu


    # Add the first layer to the dictionary
    cur_layer_num = 1
    with tf.variable_scope("Layer1"):
        weights['w' + str(cur_layer_num)] = tf.Variable(tf.random_normal([n_inputs,
                hidden_layers[0]]), name=("Layer_" + str(cur_layer_num) + "_Weights"))
        biases['b' + str(cur_layer_num)] = tf.Variable(tf.random_normal(
                [hidden_layers[0]]), name=("Layer_" + str(cur_layer_num) + "_Biases"))

    # Add all but the last layers to the dictionary
    cur_layer_num = 2
    for layer in hidden_layers[1:]:
        with tf.variable_scope("Layer" + str(cur_layer_num)):
            weights['w' + str(cur_layer_num)] = tf.Variable(tf.random_normal(
                    [hidden_layers[cur_layer_num - 2], layer]), name=("Layer_" +
                    str(cur_layer_num) + "_Weights"))
            biases['b' + str(cur_layer_num)] = tf.Variable(tf.random_normal([layer]),
                    name=("Layer_" + str(cur_layer_num) + "_Biases"))
            cur_layer_num += 1

    # Add the last layer to the dictionary
    with tf.variable_scope("OutputLayer"):
        weights['out'] = tf.Variable(tf.random_normal([hidden_layers[-1],
                n_outputs]), name="Output_Weights")
        biases['out'] = tf.Variable(tf.random_normal([n_outputs]),
                name="Output_Biases")

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

def sensitivity_and_specificity(conf):
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

    sens = conf[1][1] / (conf[1][1] + conf[1][0])
    spec = conf[0][0] / (conf[0][0] + conf[0][1])

    return sens, spec

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

def save_confusion_matrix(cm, path, classes, normalize=False,
            title='Confusion Matrix', cmap=plt.cm.Blues, name=None):

    """ This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = (cm.max() + cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Make the containing folder if not already made
    if not os.path.isdir(path):
        os.makedirs(path)

    # Save the file
    if name == None:
        plt.savefig(os.path.join(path, 'confusion_matrix.png'))
    else:
        plt.savefig(os.path.join(path, name))

    # Remove the plot from memory so it doesn't effect later plotting functions.
    plt.clf()

def numpy_long_output():
    np.set_printoptions(threshold=np.nan)

class Logger(object):
    """Logging in tensorboard without tensorflow ops.

    A class by Michael Gygli to log things which are not tensorflow ops. EG:
    just random numbers, or other things not related to the tensorflow graph.
    """

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

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
