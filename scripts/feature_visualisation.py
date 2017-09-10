import numpy as np
import os
import matplotlib.pyplot as plt


def main():

    # Get current directory and find dir or interest.
    directory = os.path.dirname(__file__)
    directory = os.path.join(directory, "..", "figs", "visu.csv")
    save_path1 = os.path.join(os.path.dirname(__file__), "..", "figs",
            "features_visualisation.png")
    save_path2 = os.path.join(os.path.dirname(__file__), "..", "figs",
            "features_visualisation_grey.png")

    # Read csv
    array = np.genfromtxt(directory, delimiter=',')

    # Plot array
    plot(np.transpose(array), save_path1, cm="jet")
    plot(np.transpose(array), save_path2, cm="Greys")


def plot(array, save_path, cm="jet"):

    conf_arr = array.tolist()

    fig = plt.figure()
    plt.clf()
    plt.title("Features over time")
    plt.xlabel("Samples")
    plt.ylabel("Feature Number")
    plt.xticks(np.arange(0, 330, 50))
    plt.yticks(np.arange(20), np.arange(1, 21))
    ax = fig.add_subplot(111)
    ax.set_aspect(10)
    if cm == "jet":
        res = ax.imshow(np.array(conf_arr), cmap=plt.cm.jet,
                interpolation='nearest', aspect="auto")
    else:
        res = ax.imshow(np.array(conf_arr), cmap=plt.cm.Greys,
                interpolation='nearest', aspect="auto")

    width = 20
    height = 100

    cb = fig.colorbar(res)

    plt.savefig(save_path, format='png')


if __name__ == "__main__":
    main()
