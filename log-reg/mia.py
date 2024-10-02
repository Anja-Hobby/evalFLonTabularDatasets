import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datasets import prepare_dataset_mia
import save
import json

FILENAME = "mia"

def train_shadow_models(data, labels, n_models=10):
    """
    Train shadow models on the data.

    Parameters:
    - data: The input data for training the models.
    - labels: The corresponding labels for the input data.
    - n_models: The number of shadow models to train (default is 10).

    Returns:
    - shadow_models: A list of tuples, where each tuple contains a trained model, X_test, and y_test.

    """
    shadow_models = []
    for _ in range(n_models):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        shadow_models.append((model, X_test, y_test))
    return shadow_models

def generate_attack_data(shadow_models):
    """
    Generate attack data for the Membership Inference Attack (MIA).

    Parameters:
    - shadow_models (list): A list of tuples containing shadow models, X_test, and y_test.

    Returns:
    - attack_dataset (list): A list of tuples containing feature vectors and attack labels.
    """

    attack_dataset = []
    for model, X_test, y_test in shadow_models:
        probabilities = model.predict_proba(X_test)
        y_test = np.array(y_test)
        for i in range(len(X_test)):
            feature_vector = np.hstack([probabilities[i], y_test[i]])
            attack_dataset.append((feature_vector, int(probabilities[i][1] > 0.5)))
    return attack_dataset

def train_attack_model(attack_data):
    """
    Trains an attack model using logistic regression.

    Parameters:
    - attack_data: A list of tuples containing the attack data. Each tuple should contain the features (X) and the corresponding labels (y).

    Returns:
    - attack_model: The trained attack model.

    """
    X_attack, y_attack = zip(*attack_data)
    attack_model = LogisticRegression()
    attack_model.fit(X_attack, y_attack)
    return attack_model

def perform_membership_inference(attack_model, target_model, data):
    """
    Perform membership inference attack on a target model using an attack model.

    Parameters:
    - attack_model (object): The attack model used to perform the membership inference attack.
    - target_model (object): The target model to be attacked.
    - data (array-like): The input data used for the attack.

    Returns:
    - inferred_membership (array-like): The inferred membership probabilities for each data point.
    """
    # No NaN values in input data
    data = data[~np.isnan(data).any(axis=1)]

    # Get the probabilities and predictions from the target model
    probabilities = target_model.predict_proba(data)
    predictions = target_model.predict(data)

    # Combine the probabilities and predictions
    attack_input = np.hstack([probabilities, predictions.reshape(-1, 1)])
    attack_input = attack_input[~np.isnan(attack_input).any(axis=1)]

    # Perform the membership inference attack
    inferred_membership = attack_model.predict_proba(attack_input)[:, 1]
    return inferred_membership

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--modeltoattack', type=str, required=True)
    parser.add_argument('--extralines', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--setupvalues', type=json.loads, default='')
    args = parser.parse_args()
    num_classes = args.num_classes
    setupvalues = args.setupvalues

    """retrieve target model"""
    print("MIA: Loading target model")

    target_model = save.load_model_params(args.location+args.modeltoattack+".npz", setupvalues)

    """retrieve data"""
    print("MIA: Loading data")
    miadatainput, miadataoutput, fldatainput, fldataoutput = prepare_dataset_mia(1)
    
    """generate attack model"""
    print("MIA: Generating attack model")
    # Train shadow model using mia-data
    shadow_models = train_shadow_models(miadatainput, miadataoutput)
    # Generate attack data
    attack_data = generate_attack_data(shadow_models)
    # Train attack model using shadow models
    attack_model = train_attack_model(attack_data)

    """perform membership inference attack"""	
    print("MIA: Performing membership inference attack")
    mia_results = perform_membership_inference(attack_model, target_model, fldatainput)

    """analyze and print results"""
    print("MIA: Analyzing and printing results")
    # analyze and print results
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
    print("Percentage of maybe-tracable records:", (len(very_likely)+len(likely)) / len(mia_results) * 100, "%")

    #save results to file
    save.save_mia_results_to_file(mia_results, very_likely, likely, unlikely, args.modeltoattack)

# How to run: python mia.py --target-model path_to_model_params.npz