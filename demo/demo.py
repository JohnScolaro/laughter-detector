import tensorflow as tf
import numpy as np
import os

# Use Matlab to turn *.wav into *.csv

# Convert *.csv into *.tfrecord

# Set up TensorFlow model

name = "demo"

# Network Params
batch_size = 500
train_test_ratio = 0.85
activation_function = 'relu'
layers = [400]
output_layer_biases = True
n_input = 20 # Data input features
n_classes = 2 # Output types. Either laughter or not laughter.
window_length = 50

start_time = time.time()
directory = os.path.dirname(__file__)
save_folder_name = helpers.get_save_dir(os.path.join(directory, 'tensorboard'), name)
pics_save_path = os.path.join(save_folder_name, 'pics')
log_save_path = os.path.join(save_folder_name, 'log')
save_save_path = os.path.join(save_folder_name, 'weights')

# Close any other local sessions if there are any.
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

dataset_file_list = []
dataset_file_list.append(os.path.join(directory, 'audio_file', '1_dataset.tfrecord'))

# Load variables from past training run

# Classify the demo audio

# Draw pictures are give stats
