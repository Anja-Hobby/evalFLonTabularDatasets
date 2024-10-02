
import flwr as fl
from collections import OrderedDict
from imblearn.over_sampling import SMOTE
from typing import Dict
import model
import read
import numpy as np
# from hydra.utils import instantiate

import torch
import flwr as fl

FILENAME = "client"
class DiabClient(fl.client.NumPyClient):
    model = []
    def __init__(self, inputdata, outputdata, initmodel, partition_id):
        """
        Initialize a client with the given input and output data.

        Args:
            inputdata (list): The input data for the client.
            outputdata (list): The output data for the client.
            initmodel: The initial model for the client.
            partition_id (int): The ID of the partition.

        Returns:
            None

        """
        self.partition_id = partition_id
        self.name = "Client " + str(partition_id)
        val_ratio = read.get_valratio_clients()
        train_ratio = 1 - val_ratio
        self.testdatainput = inputdata[:int(val_ratio*len(inputdata))]
        self.datainput= inputdata[int(train_ratio*len(inputdata)):]
        self.testdataoutput = outputdata[:int(val_ratio*len(outputdata))]
        self.dataoutput = outputdata[int(train_ratio*len(outputdata)):]
        self.model = initmodel.copy()
        self.round = 0
        self.device = torch.device("cpu")

        # balance dataset
        if read.use_smote():
            print(self.name, "Using SMOTE, oversampling data.")
            self.oversample = SMOTE()
            self.datainput, self.dataoutput = self.oversample.fit_resample(self.datainput, self.dataoutput)
        elif read.use_undersample():
            print(self.name, "Using undersampling to balance dataset.")
            self.datainput, self.dataoutput = self.remove_datapoints(self.datainput, self.dataoutput)
        else:
            print(self.name, "Using data as-is.")


    def set_parameters(self, parameters):
        """
        Update the client's model parameters. Called by the FL framework.

        Args:
            parameters (List[numpy.ndarray]): A list of updated model parameters.
        """
        print(self.name, "Setting model parameters")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        """
        Return the clients model parameters. Called by the FL framework.
            
        Returns:
            List[numpy.ndarray]: A list of numpy arrays representing the model parameters.
        """
        print(self.name, "Getting model parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config): 
        """
        Fit the model with the given parameters. Called by the FL framework.
        
        Args:
            parameters (dict): The parameters sent by the server.
            config (dict): Additional configuration for the fitting process.

        Returns:
            tuple: A tuple containing the updated parameters, the number of data inputs, and an empty dictionary.
        """
        print(self.name, "Fitting model")
        # copy the parameters sent by the server into client's local model
        self.set_parameters(parameters)
        lr = self.model.getsetupvalues().get('lr')
        momentum = self.model.get_momentum()
        epochs = self.model.get_local_epochs()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        model.train(self.model, [self.datainput, self.dataoutput], optimizer, epochs, self.device)

        return self.get_parameters({}), len(self.datainput), {}
    
    def remove_datapoints(self, datainput, dataoutput):
        """
        Removes datapoints from the input dataset to balance the classes.

        Parameters:
        - datainput (numpy.ndarray): The input dataset.
        - dataoutput (numpy.ndarray): The corresponding output labels.

        Returns:
        - balanced_datainput (numpy.ndarray): The balanced input dataset.
        - balanced_dataoutput (numpy.ndarray): The corresponding balanced output labels.
        """
        dataoutput_int = dataoutput.astype(int)
        class_counts = np.bincount(dataoutput_int)
        min_class_count = np.min(class_counts)
        indices_to_keep = []

        # Loop over each unique class label
        for class_label in np.unique(dataoutput):
            class_indices = np.where(dataoutput == class_label)[0]
            np.random.shuffle(class_indices)
            indices_to_keep.extend(class_indices[:min_class_count])

        indices_to_keep = np.array(indices_to_keep)
        np.random.shuffle(indices_to_keep)

        # Return the balanced dataset
        return datainput[indices_to_keep], dataoutput[indices_to_keep]

    def evaluate(self, parameters, config): 
        """
        Evaluates the model using the provided parameters and configuration.

        Args:
            parameters (list): The model parameters to be set.
            config (dict): The configuration settings for the evaluation.

        Returns:
            tuple: A tuple containing the loss (float), the number of test data samples (int),
                   and a dictionary of evaluation metrics ({"accuracy": float, "f1_score": float,
                   "precision": float, "recall": float}).
        """
        self.set_parameters(parameters)
        loss, accuracy, f1_score, precision, recall = model.test(self.model, [self.testdatainput, self.testdataoutput], "cpu")
        return float(loss), len(self.testdataoutput), {"accuracy": accuracy, "f1_score": f1_score, "precision": precision, "recall": recall}
        

def generate_client_fn(inputdata, outputdata, model):
    """
    Return a function that can be used by the VirtualClientEngine
    to spawn a FlowerClient with client id `cid`.

    Args:
        inputdata (list(list)): The input datas for the clients.
        outputdata (list(list)): The output datas for the clients.
        model: The model to be used by the client.

    Returns:
        function: A function that takes a client id as input and returns a FlowerClient instance.

    """
    def client_fn(cid: str):
        return DiabClient(inputdata[int(cid)], outputdata[int(cid)], model, cid).to_client()

    # return the function to spawn client
    return client_fn