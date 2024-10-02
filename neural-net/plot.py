import os

from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import save
import inspect

FILENAME = "plot"

def get_caller():
    """
    Get the name of the file that called the current function.

    Returns:
        str: The name of the calling file.
    """
    stack = inspect.stack()
    caller_frame = stack[3]
    file_name = caller_frame.filename
    file_name = file_name.split("/")[-1]
    return file_name

def visualize_conf(title, predictions, y_test):
    """
    Visualize the confusion matrix and save it as an image.

    Args:
        title (str): The title of the plot.
        predictions (array-like): The predicted labels.
        y_test (array-like): The true labels.
    """
    caller = get_caller()

    y_pred = predictions
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, cmap='Blues', annot=True, fmt='d', linewidths=5, cbar=False, annot_kws={'fontsize': 15},
                yticklabels=['Healthy', 'Not Healthy'], xticklabels=['Predicted Healthy', 'Predicted Unhealthy'],)
    plt.yticks(rotation=0)
    plt.title(title, fontsize=15)

    # Get and make the folder path
    folder_path = save.get_path()

    # save the confusion matrix
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = caller + title + current_date + '.png'
    filename = os.path.join(folder_path, filename)
    plt.savefig(filename)
    plt.close()  # Close the plot to avoid memory leaks


def plot_graphs(metrics_over_time, name):
    """
    Plot multiple metrics over rounds.

    Args:
        metrics_over_time (list): A list of dictionaries containing metrics over time.
        name (str): The name of the plot.
    """
    for key in metrics_over_time[0].keys():
        values = [metrics[key] for metrics in metrics_over_time]
        plot_over_rounds(values, key + ' over rounds', key, name)


def plot_over_rounds(accuracies, title_of_graph, y_label, name):
    """
    Plot a metric over rounds.

    Args:
        accuracies (list): A list of metric values.
        title_of_graph (str): The title of the plot.
        y_label (str): The label for the y-axis.
        name (str): The name of the plot.
    """
    title = title_of_graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('Round Number')
    plt.ylabel(y_label)
    plt.xticks(range(1, len(accuracies) + 1))  # Ensure round numbers are properly labeled
    plt.grid(True)
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = name + title + current_date + '.png'
    filename = os.path.join(save.get_path(), filename)
    plt.savefig(filename)
    plt.close()  # Close the plot to avoid memory leaks
