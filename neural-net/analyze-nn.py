
import subprocess
from datetime import datetime
import save
import datasets
from main import get_models
import read
import json
import torch
import numpy as np
from imblearn.over_sampling import SMOTE
import model as Model

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
    Run the model, save the utility results, and resulting model. Then attack the model and save the results in the same txt file.

    Args:
        model (torch.nn.Module): The neural network model to be trained and attacked.
        rounds (int): The number of rounds to train the model.
        name (str): The name of the model and results file.

    Returns:
        None (Generates txt with results and pth files with model parameters.)

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
    lr = setupvalues.get('lr')
    momentum = setupvalues.get('momentum')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    metrics_over_time = []

    # run model
    for round in range(rounds):
        # train model
        Model.train(model, [input, output], optimizer, 1, torch.device("cpu"))
        # evaluate model
        loss, accuracy, f1_score, precision, recall = Model.test(model, [serverdata[0], serverdata[1]], torch.device("cpu"))
        metrics_over_time.append({"accuracy": accuracy, "f1_score": f1_score, "precision": precision, "recall": recall, "loss": loss})

    # save results
    save.final_results(metrics_over_time, name)
    save.save_model_params(model, name)
    strategy_parameters_json = json.dumps(model.getsetupvalues())
    modelfile = name + ".pth"

    # perform MIA attack
    print(" Starting to attack model, this might take a while... ")
    print(" Model to attack: ", modelfile)
    attack = subprocess.Popen(['python3', 'mia.py', 
                               '--location', 'results/'+datetime.now().strftime("%Y-%m-%d")+'/',
                               '--setupvalues', strategy_parameters_json,
                               '--modeltoattack', modelfile])
    attack.wait()


if __name__ == "__main__":
    """
    Main entry point of the script. Input variables retrieved through the yaml, conf.yaml.
    """
    # retrieve data
    possible_rounds = read.get_numrounds()
    models = get_models() # retrieves all models to be tested

    # run models
    for rounds in possible_rounds:
        for model in models:
            name = "model_"+ datasets.get_dataset_name() + "_" + str(rounds) + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save.save_parameters(name, 0, rounds, {}, model.getsetupvalues())
            full_run(model, rounds, name)
