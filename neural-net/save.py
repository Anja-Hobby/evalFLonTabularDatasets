import os
import numpy as np
import io
from datetime import datetime
import model as Model
import torch

FILENAME = "save"

def get_path():
    """
    Get the path for saving the model and results.
    
    Returns:
        folder_path (str): The path for saving the model and results.
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
        model: The model to save.
        resultingmodelname (str): The name of the resulting model file.
    """
    print("Saving resulting model to:", resultingmodelname)
    
    # Ensure the model is on the CPU before saving
    model.cpu()
    
    resultingmodelname = resultingmodelname + ".pth"
    # Save the model's state dictionary
    model_path = os.path.join(get_path(), resultingmodelname)
    torch.save(model.state_dict(), model_path)
    print(f"Model parameters saved to {model_path}")

def load_model_params(filename, num_classes, setupvalues):
    """
    Load the model parameters from a file.
    
    Args:
        filename (str): The name of the file containing the model parameters.
        num_classes (int): The number of classes in the model.
        setupvalues: The setup values for the model.
    
    Returns:
        model: The loaded model.
    """
    print("Loading model parameters from:", filename)
    model = Model.Net(num_classes, setupvalues)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model

def ndarray_to_bytes(arr):
    """
    Convert a numpy array to bytes.
    
    Args:
        arr: The numpy array to convert.
    
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
        hist: The history to save.
    """
    folder_path = get_path()
    filename = os.path.join(folder_path, 'hist.txt')
    with open(filename, 'a') as file:
        file.write(str(hist))

def save_mia_results_to_file(mia_results, very_likely, likely, unlikely, modeltoattack):
    """
    Save the MIA (Membership Inference Attack) results to a file.
    
    Args:
        mia_results: The MIA results.
        very_likely: The records that are very likely to be in the training set.
        likely: The records that are likely to be in the training set.
        unlikely: The records that are unlikely to be in the training set.
        modeltoattack (str): The name of the model being attacked.
    """
    # Define the folder path for results
    folder_path = 'results'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = os.path.join(folder_path, datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Define the filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    percentage = (len(likely)+len(very_likely)) / len(mia_results) * 100
    name = modeltoattack[:-4]
    filename = os.path.join(folder_path, f'{name}.txt')
    
    # Open the file and write the results
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

def get_name():
    """
    Get the current timestamp as a name.
    
    Returns:
        str: The current timestamp as a name.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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