import read
import flwr as fl
from imblearn.over_sampling import SMOTE
import save

FILENAME = "client"
class DiabClient(fl.client.NumPyClient):
    """
    A client class for the Diabetes Classification model.

    Args:
        inputdata (numpy.ndarray): The input data for training and testing the model.
        outputdata (numpy.ndarray): The output data for training and testing the model.
        initmodel (object): The initial model for training.
        partition_id (int): The ID of the client partition.

    Attributes:
        partition_id (int): The ID of the client partition.
        visualize (bool): Flag indicating whether to visualize the training process.
        name (str): The name of the client.
        testdatainput (numpy.ndarray): The input data for testing the model.
        datainput (numpy.ndarray): The input data for training the model.
        testdataoutput (numpy.ndarray): The output data for testing the model.
        dataoutput (numpy.ndarray): The output data for training the model.
        oversample (object): The oversampling technique used for data augmentation.
        model (object): The current model.
        round (int): The current round of training.

    Methods:
        get_parameters(config): Get the parameters of the model.
        fit(parameters, config): Fit the model using the given parameters.
        evaluate(parameters, config): Evaluate the model using the given parameters.

    """

    model = []

    def __init__(self, inputdata, outputdata, initmodel, partition_id):
        """
        Initializes the DiabClient with data, model, and partition information.

        Args:
            inputdata (numpy.ndarray): The input data for training and testing.
            outputdata (numpy.ndarray): The output data for training and testing.
            initmodel (object): The initial model for training.
            partition_id (int): The ID of the client partition.
        """
        self.partition_id = partition_id
        self.visualize = True
        self.name = "Client " + str(partition_id)
        num_clients = save.get_num_clients()
        val_ratio = read.get_valratio_clients()
        train_ratio = 1 - val_ratio
        self.testdatainput = inputdata[:int(val_ratio*len(inputdata))]
        self.datainput= inputdata[int(train_ratio*len(inputdata)):]
        self.testdataoutput = outputdata[:int(val_ratio*len(outputdata))]
        self.dataoutput = outputdata[int(train_ratio*len(outputdata)):]

        if read.use_smote():
            print(self.name, "Using SMOTE, oversampling data.")
            self.oversample = SMOTE()
            self.datainput, self.dataoutput = self.oversample.fit_resample(self.datainput, self.dataoutput)

        if False:
            print("Data before oversampling:")
            print("Datainput:", self.datainput)
            print("Dataoutput:", self.dataoutput)
            print("Datainput shape:", self.datainput.shape)
            print("Dataoutput shape:", self.dataoutput.shape)

        self.model = initmodel.copy()
        self.round = 0

    def get_parameters(self, config):
        """
        Get the parameters of the model.

        Args:
            config (object): The configuration object.

        Returns:
            object: The parameters of the model.

        """
        return self.model.get_parameters()

    def fit(self, parameters, config):
        """
        Fit the model using the given parameters.

        Args:
            parameters (object): The parameters for training the model.
            config (object): The configuration object.

        Returns:
            tuple: A tuple containing the updated parameters, the number of training data points, and an empty dictionary.

        """
        self.model.set_parameters(parameters)
        self.model.fit(self.datainput, self.dataoutput)
        return self.model.get_parameters(), len(self.datainput), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model using the given parameters.

        Args:
            parameters (object): The parameters for evaluating the model.
            config (object): The configuration object.

        Returns:
            tuple: A tuple containing the evaluation metrics, including the accuracy, the number of test data points, and additional metrics.

        """
        self.model.set_parameters(parameters)
        a, b, c = self.model.evaluate(self.testdatainput, self.testdataoutput)
        return float(a), int(b), c

def generate_client_fn(inputdata, outputdata, model):
    """Return a function that can be used by the VirtualClientEngine.
    to spawn a FlowerClient with client id `cid`."""

    def client_fn(cid: str):
        return DiabClient(inputdata[int(cid)], outputdata[int(cid)], model, cid).to_client()

    # return the function to spawn client
    return client_fn