import numpy as np
import os
import tensorflow as tf


def writer():

    # Get the current directly
    directory = os.path.dirname(__file__)

    for file_num in range(100):

        print("Starting file number " + str(file_num))

        # Get the name of a file to convert
        name = os.path.join(directory, 'dataset/') + str(file_num) + "_dataset.csv"

        # Load the file into an array
        try:
            data = np.loadtxt(name, delimiter=',', dtype=float)
        except:
            continue

        # Create the name of the file to save our compressed data
        record_name = os.path.join(directory, 'dataset/') + str(file_num) + "_dataset.tfrecord"

        # Create a TFRecord writer
        writer = tf.python_io.TFRecordWriter(record_name)

        # Get the size of the file we are reading
        num_entries = np.shape(data)[0]
        num_features = np.shape(data)[1]

        # For every line in the file
        for entry in range(num_entries):

            # Create a training example, and populate it
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'data': tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=data[entry][0:20].tolist()
                            )
                        ),
                        'class': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(data[entry][20])]
                            )
                        ),
                        'sequence': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(data[entry][21])]
                            )
                        ),
                        'label': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=data[entry][22:24].astype(int).tolist()
                            )
                        )
                    }
                )
            )

            # Serialize it
            example_str = example.SerializeToString()

            # Write it
            writer.write(example_str)


def reader(path):

    # Open a reader
    reader = tf.python_io.tf_record_iterator(path)

    # Print everything in the file
    for example in reader:
        print(tf.train.Example().FromString(example))


def main():

    # Create the dataset
    writer()

    # Get the current directly
    #directory = os.path.dirname(__file__)
    #name = os.path.join(directory, '../../Dataset/') + str(2) + "_dataset.tfrecord"

    #reader(name)


if __name__ == '__main__':
    main()
