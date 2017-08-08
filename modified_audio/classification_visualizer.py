import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_laughter_classification(file_name, start_time=None, stop_time=None):
    """ A function which plots laughter classification times.

    This function takes a file name, start time, and a stop time and plots the
    average laughter classification values between these two times. If times are
    not specified, then it just plots the entire system.
    """

    # Get our path
    path = os.path.dirname(__file__)

    # Find largest time in all audio files
    largest_time = 0
    for time_file in get_immediate_subdirectories(path):
        time_array = np.genfromtxt(os.path.join(time_file, file_name), delimiter = ',')
        largest_time = max(time_array[-1][-1], largest_time)

    # Make huge vectors that size
    summation = np.zeros(largest_time.astype(int), dtype=np.int)
    summation_counter = 0

    for time_file in get_immediate_subdirectories(path):

        # Read the values of the .ltimes files in each subfolder
        cur_time = np.zeros(largest_time.astype(int), dtype=np.int)
        time_array = np.genfromtxt(os.path.join(time_file, file_name), delimiter = ',')

        # The numbers in the classification are set to 1, else left as 0.
        for x in range(np.shape(time_array)[0]):
            cur_time[time_array[x][0].astype(int):time_array[x][1].astype(int)] = 1

        # The cur_time is added to the summation
        summation = summation + cur_time
        summation_counter += 1

    # Normalize
    summation = summation / summation_counter

    # Plot the graph
    t = np.arange(largest_time.astype(int))

    fig = plt.figure()

    plt.plot(t, summation)
    plt.show()

def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

visualize_laughter_classification("1.ltimes")
