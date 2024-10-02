import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import read

FILENAME = "datasets"
debugging = False
class DatasetLoader:
    def __init__(self, location, target):
        """
        Initializes a new instance of the Datasets class.
        
        Args:
            location (str): The location of the dataset.
            target (str): The target variable of the dataset.
        """
        
        self.location = location
        self.target = target

    def load_data(self):
        """
        Loads the data from the specified location.

        Returns:
            pandas.DataFrame: The loaded data.
        """

        data = pd.read_csv(self.location)
        data.drop_duplicates(inplace=True)
        data.dropna(inplace=True)
        data = data.select_dtypes(include=[np.number])
        return data

    def split_data(self, data, val_ratio, method):
        """
        Splits the given data into client and server datasets based on the specified method.

        Args:
            data (pandas.DataFrame): The input data to be split.
            val_ratio (float): The ratio of validation data to be split.
            method (str): The method to be used for splitting the data. Can be 'FL' (Federated Learning) or 'MIA' (Model-Inversion Attack).

        Returns:
            tuple: A tuple containing the client input, client output, server input, and server output datasets.

        Raises:
            ValueError: If an invalid method is provided.

        """

        datalength = int(len(data) / 2)

        if method == 'FL':
            split_index = int(datalength * (1 - val_ratio))
            # Client data
            clientdata = data[:split_index]
            clientinput = clientdata.drop(self.target, axis=1)
            clientoutput = clientdata[self.target]
            # Client data
            clientdata = data[:split_index]
            clientinput = clientdata.drop(self.target, axis=1)
            clientoutput = clientdata[self.target]
            if 2 in clientoutput.values:
                clientoutput = clientoutput.map({2: 1, 1: 0})
            # Server data
            serverdata = data[split_index:datalength]
            serverinput = serverdata.drop(self.target, axis=1)
            serveroutput = serverdata[self.target]
            if 2 in serveroutput.values:
                serveroutput = serveroutput.map({2: 1, 1: 0})
            return clientinput, clientoutput, serverinput, serveroutput

        elif method == 'MIA':
            # Mia data used to train shadow models
            miadata = data[datalength:]
            miadatainput = miadata.drop(self.target, axis=1)
            miadataoutput = miadata[self.target]
            if 2 in miadataoutput.values:
                miadataoutput = miadataoutput.map({2: 1, 1: 0})
            # Server data used to validate
            fl_data = data[:datalength]
            fldatainput = fl_data.drop(self.target, axis=1)
            fldataoutput = fl_data[self.target]
            if 2 in miadataoutput.values:
                miadataoutput = miadataoutput.map({2: 1, 1: 0})
            return miadatainput, miadataoutput, fldatainput, fldataoutput

        else:
            raise ValueError("Invalid method provided. Method must be either 'FL' or 'MIA'.")

currentData = DatasetLoader(read.get_dataset_loader()[0], read.get_dataset_loader()[1]) 

def get_features():
    """
    Returns the number of features in the dataset.

    Returns:
        int: The number of features in the dataset.
    """

    data = currentData.load_data()
    return data.shape[1]-1 # Subtract 1 to exclude the target column

def get_classes():
    """
    Returns an array of unique classes from the loaded data.
    
    Returns:
        numpy.ndarray: An array of unique classes.
    """

    data = currentData.load_data()
    print("Target name : ", currentData.target)
    print("Target values: ", np.unique(data[currentData.target]))
    return np.unique(data[currentData.target])

def get_dataset_name():
    """
    Returns the name of the dataset based on the currentData location.
    
    Returns:
        str: The name of the dataset.
    """

    return currentData.location.split('/')[-1].split('.')[0]

def split_dataset(trainset, num_partitions):
    """
    Splits a dataset into multiple partitions.

    Args:
        trainset (pandas.DataFrame): The dataset to be split.
        num_partitions (int): The number of partitions to create.

    Returns:
        list: A list of pandas.DataFrame objects, each representing a partition of the dataset.
    """
    
    # Get the total number of items in the trainset
    total_size = len(trainset)
    partition_size = total_size // num_partitions
    # Determine the number of partitions that will have an extra element
    remainder = total_size % num_partitions
    trainsets = []
    start = 0
    for i in range(num_partitions):
        # Determine the end of the current partition
        end = start + partition_size + (1 if i < remainder else 0)
        # Create a subset of the dataset for the current partition
        partition = trainset.iloc[start:end]
        # Append the current partition to the result list
        trainsets.append(partition)
        # Update the start index for the next partition
        start = end
    
    return trainsets

def scale_inputdataset(inputdata):
    """
    Scales the input dataset using StandardScaler.

    Parameters:
    inputdata (pandas.DataFrame): The input dataset to be scaled.

    Returns:
    scaled_array (numpy.ndarray): The scaled input dataset.
    scaler (StandardScaler): The scaler object used for scaling.
    """

    # Initialize the scaler
    scaler = StandardScaler()
    # Fit the scaler on the input data and transform the input data
    inputarray = inputdata.to_numpy()
    scaled_array = scaler.fit_transform(inputarray)
    return scaled_array, scaler

def scale_inputdatasets(inputdatasets):
    """
    Scales a list of input datasets using the scale_inputdataset function.

    Parameters:
    inputdatasets (list): A list of input datasets to be scaled.

    Returns:
    list: A list of scaled input datasets.
    """

    return [scale_inputdataset(inputdata)[0] for inputdata in inputdatasets]

def combine_datasets(data):
    """
    Combines multiple DataFrame partitions into a single DataFrame.
    
    Args:
        data (list): A list of DataFrame partitions to be combined.
        
    Returns:
        pandas.DataFrame: The combined DataFrame.
    """

    combined_data = pd.concat(data, ignore_index=True)
    return combined_data

def prepare_dataset_fl(num_partitions: int, val_ratio: float = 0.05):
    """
    Prepare the dataset for federated learning.

    Args:
        num_partitions (int): The number of partitions to split the dataset into.
        val_ratio (float, optional): The ratio of validation data to split from the dataset. Defaults to 0.05.

    Returns:
        tuple: A tuple containing the client inputs, client outputs, and server inputs and outputs.
            - clientinputs (list): A list of numpy arrays representing the client inputs.
            - clientoutputs (list): A list of numpy arrays representing the client outputs.
            - [serverinput_np, serveroutput_np] (list): A list containing the server inputs and outputs as numpy arrays.
    """

    data = currentData.load_data()
    clientinput, clientoutput, serverinput, serveroutput = currentData.split_data(data, val_ratio, 'FL')
    # Convert clientinput and serverinput to numpy arrays after partitioning
    # Further processing
    clientinputs = split_dataset(clientinput, num_partitions)
    clientoutputs = split_dataset(clientoutput, num_partitions)
    
    # Scaling and converting to numpy arrays
    clientinputs = scale_inputdatasets(clientinputs)
    clientoutputs = [partition.to_numpy() for partition in clientoutputs]  # Convert each partition to numpy array

    serverinput_np, server_scaler = scale_inputdataset(serverinput)
    serveroutput_np = serveroutput.to_numpy()
    return clientinputs, clientoutputs, [serverinput_np, serveroutput_np]


def scale(data, scaler):
    """
    Scale the given data using the provided scaler.

    Parameters:
    data (pandas.DataFrame): The data to be scaled.
    scaler (sklearn.preprocessing.Scaler): The scaler object used for scaling.

    Returns:
    pandas.DataFrame: The scaled data.
    """

    if read.get_debugging().get(FILENAME):
        print("data: ", data)
        print("scaler: ", scaler)
    data[data.columns] = scaler.transform(data[data.columns])
    return data

def prepare_dataset_mia(num_partitions: int, val_ratio: float= 0.05):
    """
    Prepare the dataset for the MIA model.

    Args:
        num_partitions (int): The number of partitions to split the dataset into.
        val_ratio (float, optional): The ratio of validation data to be used. Defaults to 0.05.

    Returns:
        tuple: A tuple containing the prepared input and output datasets for MIA, as well as the scaled input and output datasets for FL.
    """

    data = currentData.load_data()
    miadatainput, miadataoutput, fldatainput, fldataoutput = currentData.split_data(data, val_ratio, 'MIA')

    # Scale miadatainput
    miadatainput, mia_scaler = scale_inputdataset(miadatainput)

    # Handle fldatainput partitions
    fldatainput_partitions = split_dataset(fldatainput, num_partitions)
    fldatainputs = combine_datasets(fldatainput_partitions)
    fldatainputs_scaled = scale(fldatainputs, mia_scaler)  # Ensure this returns a DataFrame

    # Handle fldataoutput if necessary
    fldataoutput_partitions = split_dataset(fldataoutput, num_partitions)
    fldataoutputs = combine_datasets(fldataoutput_partitions)

    return miadatainput, miadataoutput, fldatainputs_scaled, fldataoutputs

