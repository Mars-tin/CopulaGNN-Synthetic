"""
Visualization Utility functions
"""
import os
import sys
import matplotlib.pyplot as plt


def hold_training_plot():
    """
    Keep the program alive to display the training plot
    """
    plt.ioff()
    plt.show()


def log_training(epoch, stats):
    """
    Logs the validation accuracy and loss to the terminal
    """
    valid_loss, train_loss, _ = stats[-1]
    print('Epoch {}'.format(epoch))
    print('\tValidation Loss: {}'.format(valid_loss))
    print('\tTrain Loss: {}'.format(train_loss))


def make_training_plot(name='mlp'):
    """
    Runs the setup for an interactive matplotlib graph that logs the loss and
    accuracy
    """
    plt.ion()
    fig, axes = plt.subplots(1, figsize=(10, 5))
    plt.suptitle(name + ' Training')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')

    return fig, axes


def update_training_plot(axes, epoch, stats, log_interval):
    """
    Updates the training plot with a new data point for loss and accuracy
    """
    valid_loss = [s[0] for s in stats]
    train_loss = [s[1] for s in stats]
    axes.plot(
        range(epoch - len(stats)*log_interval + 1,
              epoch + 1, log_interval),
        valid_loss,
        linestyle='--', marker='o', color='b'
    )
    axes.plot(
        range(epoch - len(stats)*log_interval + 1,
              epoch + 1, log_interval),
        train_loss,
        linestyle='--', marker='o', color='r'
    )
    axes.legend(['Validation', 'Train'])
    plt.pause(0.00001)


def save_training_plot(fig, name='mlp'):
    """
    Saves the training plot to a file
    """
    path = "plots/"
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            sys.exit("Creation of plots directory failed.")

    path = os.path.join(path, name + '_training_plot.png')
    fig.savefig(path, dpi=200)
