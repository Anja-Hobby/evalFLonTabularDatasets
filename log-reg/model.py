import numpy as np
from sklearn.linear_model import LogisticRegression
import plot
import random
import read
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss
from diffprivlib.utils import PrivacyLeakWarning
import warnings

warnings.filterwarnings("ignore", category=PrivacyLeakWarning)

debugging = False
FILENAME = "model"

class Model:
    """
    A class representing a logistic regression model.

    Attributes:
        name (str): The name of the model.
        model (LogisticRegression): The logistic regression model.
        setupvalues (dict): The setup values for the model.

    Methods:
        __init__(**kwargs): Initializes the logistic regression model with predefined settings.
        getType(): Returns the logistic regression model.
        setup(classes, features): Sets up the model with the specified number of classes and features.
        fit(data_input, data_output): Fits the model to the provided data.
        evaluate(data_input, data_output): Evaluates the model using the provided data.
        plot(data_input, data_output): Plots the confusion matrix for the model.
        get_params(**kwargs): Returns the model parameters.
        get_parameters(): Extracts the model parameters as numpy arrays.
        set_params(**parameters): Sets the model parameters.
        set_parameters(parameters): Sets the model parameters using numpy arrays.
        get_parameters_string(): Returns a list of parameter names.
        changename(): Changes the name of the model.
        copy(): Creates a copy of the model.
        getsetupvalues(): Returns the setup values.
    """

    name = "0"

    def __init__(self, **kwargs):
        """
        Initializes the logistic regression model with predefined settings.

        Args:
            **kwargs: Additional keyword arguments to override the default parameters.

        Returns:
            None
        """
        default_params = {
            'penalty': "l1",
            'max_iter': 10,
            'solver': 'saga',
            'tol': 0.001,
            'C': 0.35,
            'warm_start': False,
            'multi_class': 'ovr'
        }
        default_params.update(kwargs)
        
        # Extract epsilon and delta from the parameters
        self.epsilon = default_params.pop('epsilon', 1.0)
        self.delta = default_params.pop('delta', 1e-5)

        if read.get_debugging().get(FILENAME):
            print("Model initialized with params:", default_params)
        
        self.setupvalues = default_params
        self.model = self.getType()
        self.model.set_params(**default_params)
        
        self.name = str(random.randint(1000,9999))

    def getType(self):
        """
        Returns the logistic regression model.

        Returns:
            LogisticRegression: The logistic regression model.
        """
        self.model = LogisticRegression()
        return self.model

    def setup(self, classes, features: int):
        """
        Sets up the model with the specified number of classes and features.

        Args:
            classes (int): The number of classes.
            features (int): The number of features.

        Returns:
            None
        """
        self.model.classes_ = np.array([0, 1])
        self.model.coef_ = np.zeros((int(classes), int(features)))
        self.model.intercept_ = np.zeros((classes,))
        print(f"Setup model", self.name, " with classes: {classes}, features: {features}")

    def fit(self, data_input, data_output):
        """
        Fits the model to the provided data.

        Args:
            data_input (array-like): The input data.
            data_output (array-like): The output data.

        Returns:
            None
        """
        print(self.name, "Fitting model", self.name)
        self.model.fit(data_input, data_output)
        
        # Adding noise for differential privacy1
        print("Adding noise for differential privacy")
        print("setupvalues:", self.setupvalues)
        noise_scale = 1.0 / self.epsilon
        self.model.coef_ += np.random.laplace(loc=0.0, scale=noise_scale, size=self.model.coef_.shape)
        self.model.intercept_ += np.random.laplace(loc=0.0, scale=noise_scale, size=self.model.intercept_.shape)
        
        # plot.visualize_conf(self.model, 'Fit Confusion Matrix', data_input, data_output)

    def evaluate(self, data_input, data_output_):
        data_output = data_output_.copy()
        """
        Evaluates the model using the provided data.

        Args:
            data_input (array-like): The input data.
            data_output (array-like): The output data.

        Returns:
            tuple: A tuple containing the loss, the number of data points, and the evaluation metrics.
        """
        print(self.name, "Evaluating model", self.name)
        y_pred = self.model.predict(data_input)
        y_pred_proba = self.model.predict_proba(data_input)
        print("data_output:", data_output)
        print("y_pred:", y_pred)
        
        if np.isnan(y_pred_proba).any():
            print("NaN values found in y_pred_proba, replacing with 0.")
            y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.0)
        if np.isnan(data_output).any():
            print("NaN values found in data_output, replacing with 0.")
            data_output = np.nan_to_num(data_output, nan=0.0)
        
        if 2 in data_input:
            data_input -= 1
        if 2 in data_output:
            data_output -= 1
        if 1 not in y_pred:
            index = np.where(y_pred == 0)[0][0]
            y_pred[index] = 1

        print("data_output:", data_output)
        print("y_pred:", y_pred)
        metrics = {"f1_score":  f1_score(data_output, y_pred, average='binary'),
                   "accuracy":   self.model.score(data_input, data_output),
                   "precision":  precision_score(data_output, y_pred, average='binary'),
                   "recall":     recall_score(data_output, y_pred, average='binary')}
        loss = log_loss(data_output,y_pred_proba)
        return loss, len(data_output), metrics
    
    def plot(self, data_input, data_output):
        """
        Plots the confusion matrix for the model.

        Args:
            data_input (array-like): The input data.
            data_output (array-like): The output data.

        Returns:
            None
        """
        plot.visualize_conf(self.model, 'Final Confusion Matrix', data_input, data_output)

    def get_params(self, **kwargs):
        """
        Returns the model parameters.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The model parameters.
        """
        if read.get_debugging().get(FILENAME):
            print(self.name, "getting model params with kwargs:", kwargs)
        return self.model.get_params(**kwargs)
    
    def get_parameters(self):
        """
        Extracts the model parameters as numpy arrays.

        Returns:
            list: A list containing the coefficient and intercept arrays.
        """
        if read.get_debugging().get(FILENAME):
            print(self.name, "Getting model parameters: Coef:", self.model.coef_.shape, "Intercept:", self.model.intercept_.shape)
        return [self.model.coef_, self.model.intercept_]
    
    def set_params(self, **parameters):
        """
        Sets the model parameters.

        Args:
            **parameters: The model parameters.

        Returns:
            None
        """
        if read.get_debugging().get(FILENAME):
            print(self.name, "Setting model params with params:", parameters)
        self.model.set_params(**parameters)

    def set_parameters(self, parameters):
        """
        Sets the model parameters using numpy arrays.

        Args:
            parameters (list): A list containing the coefficient and intercept arrays.

        Returns:
            None
        """
        if len(parameters) == 2:
            self.model.coef_, self.model.intercept_ = parameters
        else:
            raise ValueError("Expected parameters to be a list with two elements: [coef_, intercept_]")
        if read.get_debugging().get(FILENAME):
            print(self.name, "Parameters set. Coef:", self.model.coef_.shape)
            print(self.name, "Intercept:", self.model.intercept_.shape)
    
    def get_parameters_string(self):
        """
        Returns a list of parameter names.

        Returns:
            list: A list of parameter names.
        """
        parametersarray = [np.array(attr) for attr in dir(self.model) if not attr.startswith("__") and not callable(getattr(self.model, attr))]
        parameter_names = [param.item() for param in parametersarray]
        return parameter_names
    
    def changename(self):
        """
        Changes the name of the model.

        Returns:
            None
        """
        self.name = str(random.randint(1000,9999))

    def copy(self):
        """
        Creates a copy of the model.

        Returns:
            Model: A copy of the model.
        """
        new_model = Model(**self.setupvalues)
        new_model.name = self.name
        new_model.model.classes_ = self.model.classes_
        new_model.model.coef_ = self.model.coef_
        new_model.model.intercept_ = self.model.intercept_
        new_model.changename()
        return new_model
    
    def getsetupvalues(self):
        """
        Returns the setup values.

        Returns:
            dict: Setup values.
        """
        self.setupvalues['epsilon'] = self.epsilon
        self.setupvalues['delta'] = self.delta
        return self.setupvalues
    
    def predict(self, data):
        """
        Predicts the class labels for the given data.

        Parameters:
        data (array-like): The input data for prediction.

        Returns:
        array-like: The predicted class labels for the input data.
        """
        return self.model.predict(data)

    def predict_proba(self, data):
        """
        Predicts the probabilities of the target classes for the given input data.

        Parameters:
        data (array-like): The input data for which to predict the probabilities.

        Returns:
        array-like: The predicted probabilities of the target classes.
        """
        return self.model.predict_proba(data)
