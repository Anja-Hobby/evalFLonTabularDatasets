import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from model import Net
import model as netmodel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import random
import save
import json
import datasets

def train_shadow_models_nn(data, labels, setupvalues, num_features, n_models=2):
    """
    Train shadow neural network models on the data.

    Parameters:
    - data: The input data for training the models.
    - labels: The corresponding labels for the input data.
    - setupvalues: Setup values for the neural network models.
    - num_features: Number of input features for the neural network.
    - n_models: The number of shadow models to train (default is 10).

    Returns:
    - shadow_models: A list of tuples, where each tuple contains a trained model, X_test, and y_test.
    """
    shadow_models = []
    for _ in range(n_models):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5)
        model = Net(num_classes=2, setupvalues=setupvalues, num_features=num_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        netmodel.train(model, (X_train, y_train), optimizer, epochs=10, device='cpu')  # Adjust epochs and device as needed
        shadow_models.append((model, X_test, y_test))
        print("training shadow models:", random.randint(1,10))
    return shadow_models

def generate_attack_data_nn(shadow_models):
    """
    Generate attack data for the Membership Inference Attack (MIA) using neural networks.

    Parameters:
    - shadow_models (list): A list of tuples containing shadow models, X_test, and y_test.

    Returns:
    - attack_dataset (list): A list of tuples containing feature vectors and attack labels.
    """
    attack_dataset = []
    for model, X_test, y_test in shadow_models:
        model.eval()
        # Convert X_test and y_test to tensors if they are not already
        if isinstance(X_test, np.ndarray):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        if isinstance(y_test, pd.Series) or isinstance(y_test, np.ndarray):
            y_test = torch.tensor(y_test.values if isinstance(y_test, pd.Series) else y_test, dtype=torch.float32)
        
        with torch.no_grad():
            probabilities = model.predict_proba(X_test).numpy()  # Get probabilities

        for i in range(len(X_test)):
            feature_vector = np.hstack([probabilities[i], X_test[i].numpy(), y_test[i].item()])
            attack_dataset.append((feature_vector, int(probabilities[i][1] > 0.5)))
        print("generating attack data:", random.randint(1, 10))
    return attack_dataset

def train_attack_model_nn(attack_data, setupvalues):
    """
    Trains an attack model using the neural network defined in model.py.

    Parameters:
    - attack_data: A list of tuples containing the attack data. Each tuple should contain the features (X) and the corresponding labels (y).
    - setupvalues: Setup values for the neural network models.

    Returns:
    - attack_model: The trained attack neural network model.
    """
    # Prepare the attack data
    X_attack, y_attack = zip(*attack_data)
    X_attack = torch.tensor(np.array(X_attack), dtype=torch.float32)
    y_attack = torch.tensor(np.array(y_attack), dtype=torch.long)  # Use long for classification

    # Determine the number of features for the attack model
    num_features = X_attack.shape[1]  # This should match the shape of attack_input

    # Initialize the neural network model for attack
    attack_model = Net(num_classes=2, setupvalues=setupvalues, num_features=num_features)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    attack_model.train()
    epochs = 10  # Adjust number of epochs as necessary
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = attack_model(X_attack)
        loss = criterion(outputs, y_attack)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return attack_model


def perform_membership_inference_nn(attack_model, target_model, data):
    """
    Perform membership inference attack on a neural network target model using an attack neural network.

    Parameters:
    - attack_model (object): The attack model used to perform the membership inference attack.
    - target_model (object): The target neural network model to be attacked.
    - data (array-like): The input data used for the attack.

    Returns:
    - inferred_membership (array-like): The inferred membership probabilities for each data point.
    """
    # Convert data to torch tensor if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        data = torch.tensor(data.values, dtype=torch.float32)

    # Get the probabilities and predictions from the target model
    probabilities = target_model.predict_proba(data).numpy()
    predictions = target_model.predict(data).numpy()

    # Combine the probabilities, predictions, and original input data to form attack_input
    attack_input = np.hstack([probabilities, predictions.reshape(-1, 1), data.numpy()])

    # Convert attack_input to a tensor
    attack_input = torch.tensor(attack_input, dtype=torch.float32)

    # Check input size compatibility
    if attack_input.shape[1] != attack_model.fc1.in_features:
        raise ValueError(f"Input feature size mismatch: attack_input has {attack_input.shape[1]} features, but the model expects {attack_model.fc1.in_features} features.")

    # Perform the membership inference attack
    with torch.no_grad():
        inferred_membership = attack_model.predict_proba(attack_input).numpy()[:, 1]
    
    return inferred_membership





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--modeltoattack', type=str, required=True)
    parser.add_argument('--extralines', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--setupvalues', type=json.loads, default='')
    #parser.add_argument('--num_features', type=int, required=True)  # Add argument for number of features
    args = parser.parse_args()
    num_classes = args.num_classes
    setupvalues = args.setupvalues
    num_features = datasets.get_features()

    """retrieve target model"""
    print("MIA: Loading target model")
    target_model = Net(num_classes=num_classes, setupvalues=setupvalues, num_features=num_features)
    target_model.load_state_dict(torch.load(args.location + args.modeltoattack)) 

    """retrieve data"""
    print("MIA: Loading data")
    miadatainput, miadataoutput, fldatainput, fldataoutput = datasets.prepare_dataset_mia(1)

    """generate attack model"""
    print("MIA: Generating attack model")
    # Train shadow model using mia-data
    shadow_models = train_shadow_models_nn(miadatainput, miadataoutput, setupvalues, num_features)
    # Generate attack data
    attack_data = generate_attack_data_nn(shadow_models)
    # Train attack model using shadow models
    attack_model = train_attack_model_nn(attack_data, setupvalues)

    """perform membership inference attack"""	
    print("MIA: Performing membership inference attack")
    mia_results = perform_membership_inference_nn(attack_model, target_model, fldatainput)

    """analyze and print results"""
    print("MIA: Analyzing and printing results")
    very_likely = []
    likely = []
    unlikely = []

    for result in mia_results:
        if result > 0.9:
            very_likely.append(result)
        elif result > 0.5:
            likely.append(result)
        else:
            unlikely.append(result)
    print("Model information:", args.extralines)
    print("Very likely to be in the training set:", len(very_likely))
    print("Likely to be in the training set:", len(likely))
    print("Unlikely to be in the training set:", len(unlikely))
    print("Percentage of tracable records:", len(very_likely) / len(mia_results) * 100, "%")
    print("Percentage of maybe-tracable records:", (len(very_likely) + len(likely)) / len(mia_results) * 100, "%")

    # Save results to file
    save.save_mia_results_to_file(mia_results, very_likely, likely, unlikely, args.modeltoattack)
