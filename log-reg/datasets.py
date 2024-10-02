import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import read
FILENAME = "datasets"
debugging = False

class DatasetLoader:
    """
    A class for loading and splitting datasets.
    
    Attributes:
    - location: string representing the location of the dataset file.
    - target: string representing the target variable in the dataset.
    """
    
    def __init__(self, location, target):
        self.location = location
        self.target = target

    def load_data(self):
        """
        Load the dataset from the specified location.
        
        Returns:
        - data: pandas DataFrame containing the loaded dataset.
        """
        data = pd.read_csv(self.location)
        data.drop_duplicates(inplace=True)
        data.dropna(inplace=True)
        data = data.select_dtypes(include=[np.number])
        return data

    def split_data(self, data, val_ratio, method):
        """
        Split the dataset into client and server data based on the specified method.
        
        Args:
        - data: pandas DataFrame containing the dataset.
        - val_ratio: float representing the validation ratio.
        - method: string representing the splitting method ('FL' or 'MIA').
        
        Returns:
        - clientinput: pandas DataFrame containing the input features for the client data.
        - clientoutput: pandas Series containing the target values for the client data.
        - serverinput: pandas DataFrame containing the input features for the server data.
        - serveroutput: pandas Series containing the target values for the server data.
        """
        datalength = int(len(data)/2)
        
        if method == 'FL':
            split_index = int(datalength * (1 - val_ratio))
            # Client data
            clientdata = data[:split_index]
            clientinput = clientdata.drop(self.target, axis=1)
            clientoutput = clientdata[self.target]
            # Server data
            serverdata = data[split_index:datalength]
            serverinput = serverdata.drop(self.target, axis=1)
            serveroutput = serverdata[self.target]
            return clientinput, clientoutput, serverinput, serveroutput

        elif method == 'MIA':
            miadata = data[datalength:]
            miadatainput = miadata.drop(self.target, axis=1)
            miadataoutput = miadata[self.target]
            fl_data = data[:datalength]
            fldatainput = fl_data.drop(self.target, axis=1)
            fldataoutput = fl_data[self.target]
            return miadatainput, miadataoutput, fldatainput, fldataoutput

currentData = DatasetLoader(read.get_dataset_loader()[0], read.get_dataset_loader()[1])

def get_features():
    """
    Get the number of features in the dataset.
    
    Returns:
    - num_features: int representing the number of features.
    """
    data = currentData.load_data()
    return data.shape[1]-1 # Subtract 1 to exclude the target column

def get_classes():
    """
    Get the unique target values in the dataset.
    
    Returns:
    - unique_classes: numpy array containing the unique target values.
    """
    data = currentData.load_data()
    print("Target name : ", currentData.target)
    print("Target values: ", np.unique(data[currentData.target]))
    return np.unique(data[currentData.target])

def get_dataset_name():
    """
    Get the name of the dataset.
    
    Returns:
    - dataset_name: string representing the name of the dataset.
    """
    return currentData.location.split('/')[-1].split('.')[0]

def split_dataset(trainset, num_partitions):
    """
    Split the trainset into multiple partitions.
    
    Args:
    - trainset: pandas DataFrame containing the trainset.
    - num_partitions: int representing the number of partitions.
    
    Returns:
    - trainsets: list of pandas DataFrames representing the trainset partitions.
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

def split_to_train_and_test(clientsets, num_partitions, val_ratio):
    """
    Split the client datasets into train and test sets.
    
    Args:
    - clientsets: list of pandas DataFrames representing the client datasets.
    - num_partitions: int representing the number of partitions.
    - val_ratio: float representing the validation ratio.
    
    Returns:
    - trainsets: list of pandas DataFrames representing the trainsets.
    - testsets: list of pandas DataFrames representing the testsets.
    """
    trainsets = []
    testsets = []
    for clientset in clientsets:
        # Determine the split index based on the validation ratio
        split_index = int(len(clientset) * (1 - val_ratio))
        # Split the dataset into non-random train and test sets
        trainset = clientset.iloc[:split_index]
        testset = clientset.iloc[split_index:]
        # Append the train and test sets to the result lists
        trainsets.append(trainset)
        testsets.append(testset)
    return trainsets, testsets

def scale_inputdataset(inputdata):
    """
    Scale the input dataset using StandardScaler.
    
    Args:
    - inputdata: pandas DataFrame containing the input dataset.
    
    Returns:
    - scaled_array: numpy array representing the scaled input dataset.
    - scaler: StandardScaler object used for scaling.
    """
    # Initialize the scaler
    scaler = StandardScaler()
    # Fit the scaler on the input data and transform the input data
    inputarray = inputdata.to_numpy()
    scaled_array = scaler.fit_transform(inputarray)
    return scaled_array, scaler

def scale_inputdatasets(inputdatasets):
    """
    Scale a list of input datasets using StandardScaler.
    
    Args:
    - inputdatasets: list of pandas DataFrames representing the input datasets.
    
    Returns:
    - scaled_datasets: list of numpy arrays representing the scaled input datasets.
    """
    return [scale_inputdataset(inputdata)[0] for inputdata in inputdatasets]

def combine_datasets(data):
    """
    Combine a list of DataFrame partitions into a single DataFrame.
    
    Args:
    - data: list of pandas DataFrames representing the partitions.
    
    Returns:
    - combined_data: pandas DataFrame representing the combined dataset.
    """
    combined_data = pd.concat(data, ignore_index=True)
    return combined_data

def prepare_dataset_fl(num_partitions: int, val_ratio: float = 0.05):
    """
    Prepare the dataset for federated learning.
    
    Args:
    - num_partitions: int representing the number of partitions.
    - val_ratio: float representing the validation ratio.
    
    Returns:
    - clientinputs: list of numpy arrays representing the input features for the client data.
    - clientoutputs: list of numpy arrays representing the target values for the client data.
    - serverdata: list containing the input features and target values for the server data.
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
    Scale the data using the specified scaler.
    
    Args:
    - data: pandas DataFrame containing the data to be scaled.
    - scaler: StandardScaler object used for scaling.
    
    Returns:
    - scaled_data: pandas DataFrame representing the scaled data.
    """
    if read.get_debugging().get(FILENAME):
        print("data: ", data)
        print("scaler: ", scaler)
    data[data.columns] = scaler.transform(data[data.columns])
    return data

def prepare_dataset_mia(num_partitions: int, val_ratio: float= 0.05):
    """
    Prepare the dataset for model inversion attack.
    
    Args:
    - num_partitions: int representing the number of partitions.
    - val_ratio: float representing the validation ratio.
    
    Returns:
    - miadatainput: pandas DataFrame containing the input features for the MIA data.
    - miadataoutput: pandas Series containing the target values for the MIA data.
    - fldatainputs_scaled: pandas DataFrame representing the scaled input features for the FL data.
    - fldataoutputs: pandas Series containing the target values for the FL data.
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