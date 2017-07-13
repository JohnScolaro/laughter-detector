import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def laughter_plotter(prediction, label, time_step):
    """ Plots predictions and labels visually.

    Inputs:
        prediction: The numpy 2D array of laughter and/or not laughter values.
        label: The numpy 2D array of labels
        time_step: The time step between records in seconds.

    This function plots the predictions and labels against eachother, so we can
    see how accurate our function actually is visually.
    """

    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()
