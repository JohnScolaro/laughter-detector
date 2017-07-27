import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
import os

def laughter_plotter(prediction, label, path, number, time_step, batch_size):
    """ Plots predictions and labels visually.

    Inputs:
        prediction: The numpy 2D array of laughter and/or not laughter values.
        label: The numpy 2D array of labels
        path: A path to the folder you want to store the images in.
        number: Any number to be appended to the end of the image.
        time_step: The time step between records in seconds.

    This function plots the predictions and labels against eachother, so we can
    see how accurate our function actually is visually.
    """
    prediction = np.transpose(prediction)[1]
    label = np.transpose(label)[1]

    # Create 5000 by 1 vector of times
    times = np.linspace(0.0, float(batch_size * time_step), num=batch_size, endpoint=False)

    fig = plt.figure(figsize=(15, 5))

    # Plot the laughter predictions.
    plt.plot(times[0:500], prediction[0:500], 'b', linewidth=1)

    # Plot the labels.
    plt.plot(times[0:500], label[0:500], 'k', linewidth=2)

    # Do custom pretty things like axis labels, legend, and colours.
    plt.xlabel('Time (s)')
    plt.ylabel('Probability (%)')
    blue_patch = mpatches.Patch(color='blue', label='Probability of Laughter')
    black_patch = mpatches.Patch(color='black', label='Data Labels')
    plt.legend(handles=[black_patch, blue_patch], loc=1)
    plt.axis([0, 10, 0, 1])

    # If the sound folder doesn't exist, make it.
    path = os.path.join(path, 'sound')
    if not os.path.isdir(path):
        os.makedirs(path)

    # Save plot, and clear from memory.
    plt.savefig(os.path.join(path, 'classification' + str(number) + '.png'))
    plt.clf()
    plt.close(fig)

def save_confusion_matrix(cm, path, classes, normalize=False,
            title='Confusion Matrix', cmap=plt.cm.Blues, name=None):

    """ This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    fig = plt.figure(figsize=(6, 5))
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

    # Make the conf container folder.
    path = os.path.join(path, 'conf')
    if not os.path.isdir(path):
        os.makedirs(path)

    # Save the file
    if name == None:
        plt.savefig(os.path.join(path, 'confusion_matrix.png'))
    else:
        plt.savefig(os.path.join(path, name))

    # Remove the plot from memory so it doesn't effect later plotting functions.
    plt.clf()
    plt.close(fig)
