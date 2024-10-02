import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import datasets
import plot
from flwr.common.parameter import ndarrays_to_parameters
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

FILENAME = "model"

class Net(nn.Module):
    """
    Neural network model class.

    Args:
        num_classes (int): Number of output classes.
        setupvalues (dict): Dictionary containing setup values for the model.

    Attributes:
        name (str): Name of the neural network model.
        num_classes (int): Number of output classes.
        setupvalues (dict): Dictionary containing setup values for the model.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer (optional).
        fc4 (nn.Linear): Fourth fully connected layer (optional).
        fc5 (nn.Linear): Fifth fully connected layer (optional).

    Methods:
        forward(x): Performs forward pass through the neural network.
        predict_proba(x): Predicts class probabilities for the input tensor.
        predict(x): Predicts class labels for the input tensor.
        get_momentum(): Returns the momentum value from the setup values.
        get_local_epochs(): Returns the number of local epochs from the setup values.
        copy(): Creates a copy of the neural network model.
        setname(): Sets the name of the neural network model.
        get_params(): Returns the model parameters.
        get_parameters_string(): Returns the setup values as a string.
        get_parameters(): Returns the model parameters.
        plot(data_input, data_output): Plots the confusion matrix.
        getsetupvalues(): Returns the setup values.
    """

    name = "NN"

    def __init__(self, num_classes: int, setupvalues, num_features: int) -> None:
        """
        Initializes the neural network model.

        Args:
            num_classes (int): Number of output classes.
            setupvalues (dict): Dictionary containing setup values for the model.
        """
        self.setupvalues = setupvalues
        # num_features = datasets.get_features()
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 2*num_features)
        self.fc2 = nn.Linear(2*num_features, 2*num_features)
        if setupvalues.get('layers') == 3:
            self.fc3 = nn.Linear(2*num_features, num_classes)
        if setupvalues.get('layers') == 4:
            self.fc3 = nn.Linear(2*num_features, 2*num_features)
            self.fc4 = nn.Linear(2*num_features, num_classes)
        if setupvalues.get('layers') == 5:
            self.fc3 = nn.Linear(2*num_features, 2*num_features)
            self.fc4 = nn.Linear(2*num_features, 2*num_features)
            self.fc5 = nn.Linear(2*num_features, num_classes)
        self.setname()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.setupvalues.get('layers') == 3:
            return self.fc3(x)
        if self.setupvalues.get('layers') == 4:
            x = F.relu(self.fc3(x))
            return self.fc4(x)
        if self.setupvalues.get('layers') == 5:
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            return self.fc5(x)
        x = self.fc3(x)
        return x

    def predict_proba(self, x):
        """
        Predicts class probabilities for the input tensor.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Predicted class probabilities.
        """
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities

    def predict(self, x):
        """
        Predicts class labels for the input tensor.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Predicted class labels.
        """
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def get_momentum(self): 
        """
        Returns the momentum value from the setup values.

        Returns:
            float: Momentum value.
        """
        return self.setupvalues.get('momentum')
    
    def get_local_epochs(self):
        """
        Returns the number of local epochs from the setup values.

        Returns:
            int: Number of local epochs.
        """
        return self.setupvalues.get('local_epochs')
    
    def copy(self):
        """
        Creates a copy of the neural network model.

        Returns:
            Net: Copied neural network model.
        """
        num_features = datasets.get_features()
        return Net(self.num_classes, self.setupvalues, num_features)
    
    def setname(self):
        """Sets the name of the neural network model."""
        self.name = "NN" + str(random.randint(1000, 9999))

    def get_params(self):
        """
        Returns the model parameters.

        Returns:
            dict: Model parameters.
        """
        return model_to_parameters(self)
    
    def get_parameters_string(self):
        """
        Returns the setup values as a string.

        Returns:
            str: Setup values as a string.
        """
        return self.setupvalues
    
    def get_parameters(self):
        """
        Returns the model parameters.

        Returns:
            torch.nn.parameter.Parameter: Model parameters.
        """
        return self.model.parameters()
    
    def plot(self, data_input, data_output):
        """
        Plots the confusion matrix.

        Args:
            data_input: Input data.
            data_output: Output data.
        """
        input = pd.DataFrame(data_input)
        data_tensor = torch.tensor(input.values).float()
        self.eval()
        with torch.no_grad():
            predictions = self(data_tensor)
            predicted_classes = torch.argmax(predictions, dim=1)
        
        predictions = predicted_classes.numpy()
        # plot.visualize_conf( 'Final Confusion Matrix', predictions , data_output)

    def getsetupvalues(self):
        """
        Returns the setup values.

        Returns:
            dict: Setup values.
        """
        return self.setupvalues

def train(net, traindata, optimizer, epochs, device: str):
    """
    Trains the neural network model on the training set.

    Args:
        net (Net): Neural network model.
        traindata: Training data.
        optimizer: Optimizer for training.
        epochs (int): Number of training epochs.
        device (str): Device to use for training.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)

    inputdata = traindata[0]
    labels = traindata[1]
    trainloader = torch.utils.data.DataLoader(list(zip(inputdata, labels)), batch_size=net.setupvalues.get('batch_size'), shuffle=True)
    setupvalues = net.setupvalues
    
    for _ in range(epochs):
        for data, labels in trainloader:

            data, labels = data.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            loss = criterion(net(data), labels)
            loss.backward()

            # Start DP 
            if net.setupvalues.get('use_dp'):
                # clip gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), setupvalues.get('max_grad_norm'))
                # Add noise to the gradients
                for param in net.parameters():
                    if param.grad is not None:
                        noise = torch.normal(0, setupvalues.get('noise_multiplier') * setupvalues.get('max_grad_norm'), size=param.grad.size(), device = param.grad.device)
                        param.grad += noise
                # End DP

            optimizer.step()

def test(net, testdata, device:str):
    """
    Validates the neural network model on the test set.

    Args:
        net (Net): Neural network model.
        testdata: Test data.
        device (str): Device to use for evaluation.

    Returns:
        tuple: Tuple containing loss, accuracy, F1 score, precision, and recall.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    inputdata = testdata[0]
    datalabels = testdata[1]
    testloader = torch.utils.data.DataLoader(list(zip(inputdata, datalabels)), batch_size=1)
    predictedoutput=[]
    with torch.no_grad():
        for data in testloader:
            input = data[0].clone().detach().to(device, dtype=torch.float32)
            labels = data[1].clone().detach().to(device, dtype=torch.long)
            outputs = net(input)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            predictedoutput.append(predicted.item())
    accuracy = correct / len(testloader.dataset)
    f1score = f1_score(datalabels, predictedoutput, average='binary')
    precision = precision_score(datalabels, predictedoutput, average='binary')
    recall = recall_score(datalabels, predictedoutput, average='binary')
    return loss, accuracy, f1score, precision, recall

def model_to_parameters(model):
    """
    Extracts the model parameters.

    Args:
        model: Neural network model.

    Returns:
        flwr.common.parameter.Parameters: Model parameters.
    """
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)
    return parameters