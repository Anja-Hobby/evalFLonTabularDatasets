import argparse
from imblearn.over_sampling import SMOTE
import subprocess
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
import mia
import model as Model

import numpy as np
import datasets
import read
from main import get_models
import save
import json

selected_model = LogisticRegression()

def remove_datapoints(datainput, dataoutput):
    """
    Remove datapoints from the dataset to balance the classes.

    Parameters:
    - datainput (numpy.ndarray): Input data array.
    - dataoutput (numpy.ndarray): Output data array.
    
    Returns:
    - balanced_datainput (numpy.ndarray): Balanced input data array.
    - balanced_dataoutput (numpy.ndarray): Balanced output data array.
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

def full_run(model, rounds, name):
    """
    Run the full analysis pipeline for the logistic regression model.

    Args:
        model: The logistic regression model object.
        rounds: The number of rounds to run the analysis.
        name: The name of the analysis.

    Returns:
        None
    """
    # prepare data
    clientinputs, clientoutputs, serverdata = datasets.prepare_dataset_fl(1, 0.1)
    input = clientinputs[0]
    output = clientoutputs[0]
    # balance dataset
    if read.use_smote():
        oversample = SMOTE()
        input, output = oversample.fit_resample(input, output)
    elif read.use_undersample():
        input, output = remove_datapoints(input, output)

    # prepare model
    setupvalues = model.getsetupvalues()
    log_reg = Model.Model(**setupvalues)
    metrics_over_time = []
    for round in range(rounds):
        # train model
        log_reg.fit(input, output)
        # evaluate model
        # Predict and evaluate model
        y_pred = log_reg.predict(serverdata[0])
        y_pred_proba = log_reg.predict_proba(serverdata[0])
        
        accuracy = accuracy_score(serverdata[1], y_pred)
        f1 = f1_score(serverdata[1], y_pred, average='weighted')
        precision = precision_score(serverdata[1], y_pred, average='weighted')
        recall = recall_score(serverdata[1], y_pred, average='weighted')
        loss = log_loss(serverdata[1], y_pred_proba)
        
        metrics_over_time.append({"accuracy": accuracy, "f1_score": f1, "precision": precision, "recall": recall, "loss": loss})

    # save results
    save.final_results(metrics_over_time, name)
    save.save_model_params(model, name)
    strategy_parameters_json = json.dumps(model.getsetupvalues())
    modelfile = name + ".npz"

    # perform MIA attack
    print(" Starting to attack model, this might take a while... ")
    print(" Model to attack: ", modelfile)
    mia.attack_model(log_reg, name)


if __name__ == "__main__":
    """
    Main entry point of the script. Input variables retrieved through the yaml, conf.yaml.
    """
    # retrieve data
    possible_rounds = read.get_numrounds()
    models = get_models() # retrieves all models to be tested
    parser = argparse.ArgumentParser()

    # run models
    for rounds in possible_rounds:
        for model in models:
            name = "model_"+ datasets.get_dataset_name() + "_" + str(rounds) + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save.save_parameters(name, 0, rounds, {}, model.getsetupvalues())
            full_run(model, rounds, name)
