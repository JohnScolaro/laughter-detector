import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
import os
from sklearn import metrics

def multiple_laughter_plotter(sess, test_label, soft_mlp_test, test_iter,
        pics_save_path, batch_size, window_length):

    """ A function which handles creating all the classification pictures.

    This function creates all the classification pictures after training has
    concluded. It has been moved to one function in order to simplify the code
    in the main network file.
    """

    # Reinitialize the testing dataset handle.
    sess.run(test_iter.initializer)

    # Create a bunch of plots.
    for x in range(50):
        try:
            lab, pred = sess.run([test_label, soft_mlp_test])

            #TODO: Remove a when you're done debugging this.
            laughter_plotter(pred, lab, pics_save_path, x, 0.02,
                    batch_size, window_length)

        except:
            pass

def laughter_plotter(prediction, label, path, number, time_step, batch_size,
        window_length):
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
    plot_length = 500
    if batch_size < 600:
        plot_length = batch_size - window_length - 1

    plt.plot(times[0:plot_length], prediction[0:plot_length], 'b', linewidth=1)

    # Plot the labels.
    plt.plot(times[0:plot_length], label[0:plot_length], 'k', linewidth=2)

    # Do custom pretty things like axis labels, legend, and colours.
    plt.xlabel('Time (s)')
    plt.ylabel('Probability (%)')
    blue_patch = mpatches.Patch(color='blue', label='Probability of Laughter')
    black_patch = mpatches.Patch(color='black', label='Data Labels')
    plt.legend(handles=[black_patch, blue_patch], loc=1)
    plt.axis([0, 10, -0.1, 1.2])

    # If the sound folder doesn't exist, make it.
    path = os.path.join(path, 'sound')
    if not os.path.isdir(path):
        os.makedirs(path)

    # Save plot, and clear from memory.
    plt.savefig(os.path.join(path, 'classification' + str(number) + '.png'))
    plt.clf()
    plt.close(fig)

def save_final_confusion_matrixes(conf, pics_save_path):
    """ Saves a normalized and un-normalized confusion matrix.

    Saves a normalized and un-normalized confusion matrix. It is done in a
    seperate file in order to save space in the main network file.
    """

    save_confusion_matrix(conf, pics_save_path,
            classes=['Not Laughter', 'Laughter'],
            name='final_confusion_matrix')
    save_confusion_matrix(conf, pics_save_path,
            classes=['Not Laughter', 'Laughter'], normalize=True,
            name='final_norm_confusion_matrix')

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

def multiple_roc_curve_plotter(sess, test_label, soft_mlp_test, test_iter,
        pics_save_path):

    """ Creates the data to plot the ROC curve.

    Moved into a seperate function in order to reduce the amount of code in the
    actualy sequence network file, this program creates ROC curves for the
    classified data and saves it as a picture.
    """

    sess.run(test_iter.initializer)
    concat_lab = np.array([])
    concat_pred = np.array([])

    flag = False
    while(1):
        try:
            lab, pred = sess.run([test_label, soft_mlp_test])
            if flag == False:
                concat_lab = lab
                concat_pred = pred
                flag = True
            concat_lab = np.concatenate((concat_lab, lab), axis=0)
            concat_pred = np.concatenate((concat_pred, pred), axis=0)
        except tf.errors.OutOfRangeError:
            break

    roc_curve_plotter(concat_pred[:,0], concat_lab[:,0], pics_save_path, name='laughter.png')
    roc_curve_plotter(concat_pred[:,1], concat_lab[:,1], pics_save_path, name='non_laughter.png')

def roc_curve_plotter(prediction, label, path=None, name=None):
    """ Computes and plots the ROC curve for the test dataset.

    """

    fpr, tpr, thresholds = metrics.roc_curve(label, prediction, pos_label=1)

    # Plot of a ROC curve for a specific class
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Laughter')
    plt.legend(loc="lower right")

    if path == None:
        plt.show()
    else:
        # Make the containing folder if not already made
        if not os.path.isdir(path):
            os.makedirs(path)

        # Make the conf container folder.
        path = os.path.join(path, 'roc')
        if not os.path.isdir(path):
            os.makedirs(path)

        # Save the file
        if name == None:
            plt.savefig(os.path.join(path, 'roc_curve.png'))
        else:
            plt.savefig(os.path.join(path, name))

        # Remove the plot from memory so it doesn't effect later plotting functions.
        plt.clf()
        plt.close(fig)
