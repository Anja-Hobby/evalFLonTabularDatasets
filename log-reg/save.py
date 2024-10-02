import os
import numpy as np
import io
import model as Model
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import re
FILENAME = "save"

def get_path():
    """
    Get the path for saving the files.

    Returns:
        str: The folder path for saving the files.
    """
    folder_path = 'results'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = os.path.join(folder_path, datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def save_model_params(model, resultingmodelname):
    """
    Save the model parameters to a file.

    Args:
        model: The trained model.
        resultingmodelname (str): The name of the resulting model file.
    """
    parameters = {'coefficients': model.get_parameters()[0], 'intercept': model.get_parameters()[1]}
    np.savez(os.path.join(get_path(), resultingmodelname), **parameters)

def load_model_params(filename, setupvalues):
    """
    Load the model parameters from a file.

    Args:
        filename (str): The name of the model file.
        setupvalues: The setup values for the model.

    Returns:
        Model: The loaded model.
    """
    data = np.load(filename)
    model = Model.Model(**setupvalues)
    model.model.coef_ = data['coefficients']
    model.model.intercept_ = data['intercept']
    model.model.classes_ = np.array([0, 1])

    # Set required attributes for scikit-learn's LogisticRegression
    model.model.n_iter_ = np.array([setupvalues.get('max_iter', 10)])  # Setting this as a placeholder
    model.model.n_features_in_ = model.model.coef_.shape[1]
    if hasattr(model, '_is_fitted'):
        model._is_fitted = True
    
    return model

def ndarray_to_bytes(arr):
    """
    Convert a numpy array to bytes.

    Args:
        arr: The numpy array.

    Returns:
        bytes: The converted bytes.
    """
    bytes_io = io.BytesIO()
    np.save(bytes_io, arr, allow_pickle=False)
    return bytes_io.getvalue()

def save_num_clients(num_clients):
    """
    Save the number of clients to a text file.

    Args:
        num_clients (int): The number of clients.
    """
    with open('num_clients.txt', 'w') as file:
        file.write(str(num_clients))

def get_num_clients():
    """
    Get the number of clients from a text file.

    Returns:
        int: The number of clients.
    """
    with open('num_clients.txt', 'r') as file:
        return int(file.read())

def save_to_text(hist):
    """
    Save the history to a text file.

    Args:
        hist: The history to be saved.
    """
    folder_path = get_path()
    filename = os.path.join(folder_path, 'hist.txt')
    with open(filename, 'a') as file:
        file.write(str(hist))


def extract_date_from_name(name):
    """
    Extract the year, month, and day from the name string.

    Args:
        name (str): The name of the file that contains the year, month, and day.

    Returns:
        folder_path (str): A string representing the folder path derived from the year, month, and day.
    """
    # Regular expression to find the 4-digit year
    year_match = re.search(r'\d{4}', name)
    
    if year_match:
        year = year_match.group(0)  # Get the first 4-digit number (assumed to be the year)
        # Extract the month and day after the year (next two 2-digit numbers)
        remaining_string = name[year_match.end():]
        month_day_match = re.search(r'(\d{2})-(\d{2})', remaining_string)
        
        if month_day_match:
            month = month_day_match.group(1)
            day = month_day_match.group(2)
            # Construct the folder path in the format 'year-month-day'
            formatted_date = f"{year}-{month}-{day}"
            folder_path = os.path.join('results', formatted_date)
            return folder_path
        else:
            raise ValueError("Could not find valid month and day in the name.")
    else:
        raise ValueError("Could not find valid year in the name.")


def save_mia_results_to_file(mia_results, very_likely, likely, unlikely, name):
    """
    Save the MIA (Membership Inference Attack) results to a file.

    Args:
        mia_results: The MIA results.
        very_likely: The records that are very likely to be in the training set.
        likely: The records that are likely to be in the training set.
        unlikely: The records that are unlikely to be in the training set.
        name (str): The name of the file.
    """
    folder_path = 'results'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    folder_path = extract_date_from_name(name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    percentage = (len(very_likely)+len(likely)) / len(mia_results) * 100
    filename = os.path.join(folder_path, f'{name}.txt')
    with open(filename, 'a') as file:
        file.write(f"timestamp: {current_time}\n")
        file.write(f"Very likely to be in the training set (90%+): {len(very_likely)}\n")
        file.write(f"Likely to be in the training set (50-90%): {len(likely)}\n")
        file.write(f"Unlikely to be in the training set (50%-): {len(unlikely)}\n")
        file.write(f"Percentage of traceable records (90%+): {len(very_likely) / len(mia_results) * 100:.2f}%\n")
        file.write(f"Percentage of maybe-traceable records (50%+): {percentage:.2f}%\n")
        file.write(f"100 of the results: ")
        for result in mia_results[:100]:
            file.write(f"\n{result}")

def save_parameters(name, num_clients, rounds, strategy_parameters, model_parameters):
    """
    Save the parameters to a text file.

    Args:
        name (str): The name of the file.
        num_clients (int): The number of clients.
        rounds (int): The number of rounds.
        strategy_parameters: The strategy parameters.
        model_parameters: The model parameters.
    """
    folder_path = get_path()
    name = name + '.txt'
    filename = os.path.join(folder_path, name)
    with open(filename, 'w') as file:
        file.write(f"Number of clients: {num_clients}\n")
        file.write(f"Number of rounds: {rounds}\n")
        file.write(f"Strategy parameters: {strategy_parameters}\n")
        file.write(f"Model parameters: {model_parameters}\n")

def final_results(metrics_over_time, name):
    """
    Save the final results to a text file.

    Args:
        metrics_over_time: The metrics over time.
        name (str): The name of the file.
    """
    folder_path = get_path()
    filename = os.path.join(folder_path, name + '.txt')
    with open(filename, 'a') as file:
        for metrics in metrics_over_time:
            file.write(f"{metrics}\n")