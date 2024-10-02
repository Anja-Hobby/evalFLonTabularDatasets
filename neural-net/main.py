import subprocess
from datetime import datetime
import flwr as fl
from datasets import prepare_dataset_fl, get_dataset_name, get_features, get_classes
from client import generate_client_fn
import save
from server import get_on_fit_config, get_evaluate_fn
from omegaconf import OmegaConf
from model import Net
import numpy as np
import read
import itertools
import time
import json


debugging = read.get_debugging().get("main")
valratio = 0.05
FILENAME = "main"

def get_strategy(model, numrounds, strategycombo, name):
    """
    Get the federated learning strategy based on the given parameters.

    Args:
        model: The neural network model.
        numrounds: The number of communication rounds.
        strategycombo: The combination of strategy parameters.
        name: The name of the strategy.

    Returns:
        The federated learning strategy.

    Raises:
        ValueError: If the strategy name is unknown.
    """
    strategyname = strategycombo.get('name')
    _, _, serverdata = prepare_dataset_fl(1, valratio)
    if strategyname == 'FedAvg': # based on https://arxiv.org/abs/1602.05629
        # TODO: moeten deze parameters hier wel gegeven worden? (lr, momentum, local_epochs)
        return fl.server.strategy.FedAvg(on_fit_config_fn=get_on_fit_config(OmegaConf.create({"lr":strategycombo.get('lr'), 
                                                                                           "momentum":strategycombo.get('momentum'), 
                                                                                           "local_epochs":strategycombo.get('local_epochs')})), 
                                          evaluate_fn=get_evaluate_fn(model, serverdata ,numrounds, name)
                                          )
    elif strategyname == 'FedAdagrad': # based on https://arxiv.org/abs/2003.00295v5
        # TODO: check wether eta_l is should be passed here
        return fl.server.strategy.FedAdagrad(eta = strategycombo.get('eta'),
                                            eta_l = strategycombo.get('eta_l'),
                                            tau = strategycombo.get('tau'),
                                            evaluate_fn=get_evaluate_fn(model, serverdata ,numrounds, name),
                                            initial_parameters=model.get_params(),
                                            )
    elif strategyname == 'FedAdam': # based on https://arxiv.org/abs/2003.00295v5
        return fl.server.strategy.FedAdam(
            eta=strategycombo.get('eta'), 
            eta_l=strategycombo.get('eta_l'),
            beta_1=strategycombo.get('beta_1'), 
            beta_2=strategycombo.get('beta_2'), 
            tau=strategycombo.get('tau'),
            evaluate_fn=get_evaluate_fn(model, serverdata ,numrounds, name),        
            initial_parameters=model.get_params()
            )
    elif strategyname == 'FedYogi': # based on https://arxiv.org/abs/2003.00295v5
        return fl.server.strategy.FedYogi(
            eta=strategycombo.get('eta'),
            eta_l=strategycombo.get('eta_l'),
            beta_1=strategycombo.get('beta_1'),
            beta_2=strategycombo.get('beta_2'),
            tau=strategycombo.get('tau'),
            evaluate_fn=get_evaluate_fn(model, serverdata ,numrounds, name),  
            initial_parameters=model.get_params()
        )
    elif strategyname == 'FedAvgM': # based on https://arxiv.org/abs/1909.06335
        return fl.server.strategy.FedAvgM(
            server_learning_rate=strategycombo.get('server_learning_rate'),
            server_momentum=strategycombo.get('server_momentum'),
            evaluate_fn=get_evaluate_fn(model, serverdata ,numrounds, name),  
            initial_parameters=model.get_params()             
        )
    elif strategyname == 'FedMedian': #TODO add source
        return fl.server.strategy.FedMedian(
            evaluate_fn=get_evaluate_fn(model, serverdata ,numrounds, name),  
            initial_parameters=model.get_params(),
        ) # has no parameters
    elif strategyname == 'FedProx': # based on https://arxiv.org/abs/1812.06127
        #TODO: see https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedProx.html 
        # proximal term needs to be added to the loss function during training on client.
        return fl.server.strategy.FedProx(
            proximal_mu=strategycombo.get('proximal_mu'),
            evaluate_fn=get_evaluate_fn(model, serverdata ,numrounds, name),  
            initial_parameters=model.get_params()
        )
    elif strategyname == 'QFedAvg': #TODO find out wether we want to even try this.
        return fl.server.strategy.QFedAvg(
            evaluate_fn=get_evaluate_fn(model, serverdata ,numrounds, name),  
            initial_parameters=model.get_params()
        )
    elif strategyname == 'FedTrimmedAvg': # based on: https://arxiv.org/abs/1803.01498
        return fl.server.strategy.FedTrimmedAvg(
            evaluate_fn=get_evaluate_fn(model, serverdata ,numrounds, name),  
            beta=strategycombo.get('beta'),
            initial_parameters=model.get_params()
        )
    elif strategyname == 'FedOpt': # based on https://arxiv.org/abs/2003.00295v5
        return fl.server.strategy.FedOpt(
            evaluate_fn=get_evaluate_fn(model, serverdata ,numrounds, name),  
            eta=strategycombo.get('eta'), 
            eta_l=strategycombo.get('eta_l'), 
            beta_1=strategycombo.get('beta_1'), 
            beta_2=strategycombo.get('beta_2'), 
            tau=strategycombo.get('tau'),
            initial_parameters=model.get_params()
        )
    else:
        if debugging:
            print("Information about strategy: ", strategycombo)
        raise ValueError("Unknown strategy: " + strategyname)


def run_federated_learning(model, num_clients, rounds, strategy, name):
    """
    Run federated learning simulation.

    Args:
        model: The neural network model.
        num_clients: The number of clients.
        rounds: The number of communication rounds.
        strategy: The federated learning strategy.
        name: The name of the simulation.

    Returns:
        None
    """
    if read.get_debugging().get(FILENAME): print("Main: Starting federated learning")
    clientinputs, clientoutputs, serverdata = prepare_dataset_fl(num_clients, valratio)
    classes = np.unique(serverdata[1])
    if read.get_debugging().get(FILENAME):
        print("Generating client_fn with following parameters: ")
        print("Client input: ", clientinputs)
        print("Client output: ", clientoutputs)
        print("Model: ", model)
    client_fn = generate_client_fn(clientinputs, clientoutputs, model)
    
    #full_strategy = instatiate(strategy, evaluate_fn=get_evaluate_fn(model, testsets))
    #name = "model_"+ get_dataset_name() + "_" + str(rounds) + "_" + read.get_strategy().get('name') + "_" + str(num_clients) + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_strategy = strategy
    
    if read.get_debugging().get(FILENAME): 
        print("Main: Starting simulation with following parameters: ")
        print("Main: Num_clients: ", num_clients)
        print("Main: Config: ", fl.server.ServerConfig(num_rounds=rounds))
        print("Main: Strategy: ", full_strategy)
        print("Main: Ray_init_args: ", {'num_cpus': 2, 'num_gpus': 0})

    hist = fl.simulation.start_simulation(
        client_fn = client_fn,
        num_clients = num_clients,
        config = fl.server.ServerConfig(num_rounds=rounds),
        strategy = full_strategy,
        ray_init_args = {'num_cpus': 2, 'num_gpus': 0},
    )

    save.save_to_text(hist)


def full_single_run(model, num_clients, rounds, strategy, name):
    """
    Run a full single federated learning run and perform model attack.

    Args:
        model: The neural network model.
        num_clients: The number of clients.
        rounds: The number of communication rounds.
        strategy: The federated learning strategy.
        name: The name of the simulation.

    Returns:
        None
    """
    run_federated_learning(model, num_clients, rounds, strategy, name)
    strategy_parameters_json=json.dumps(model.getsetupvalues())
    modelfile = name + ".pth"
    print(" Starting to attack model, this might take a while... ")
    print(" Model to attack: ", modelfile)
    attack = subprocess.Popen(['python3', 'mia.py', 
                               '--location', 'results/'+datetime.now().strftime("%Y-%m-%d")+'/',
                               '--setupvalues', strategy_parameters_json,
                               '--modeltoattack', modelfile])
    attack.wait()


def get_models():
    """
    Get a list of neural network models.

    Returns:
        A list of neural network models.
    """
    print("Getting models")
    classes = len(get_classes())
    features = get_features()
    if read.get_debugging().get(FILENAME):
        print("Classes: ", classes)
        print("Features: ", features)
    modeldict = read.model_dictionary()
    keys, values = zip(*modeldict.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    models = []
    num_features = get_features()

    for setupvalue in combinations:
        model = Net(classes, setupvalue, num_features)
        models.append(model)

    return models


def get_strategy_combos():
    """
    Get a list of strategy combinations.

    Returns:
        A list of strategy combinations.
    """
    print("Getting strategies")
    strategiesdict = read.get_strategy()
    keys, values = zip(*strategiesdict.items())
    
    # Ensure each value in values is a list, to avoid splitting strings
    values = [v if isinstance(v, list) else [v] for v in values]
    
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    strategies = []
    for combo in combinations:
        strategies.append(combo)
    
    if read.get_debugging().get(FILENAME):
        print("Strategies: ", strategies)
    
    return strategies


def full_gridsearch_run(models, num_clients_array, rounds_array, strategy_combos):
    """
    Run a full grid search for federated learning.

    Args:
        models: A list of neural network models.
        num_clients_array: An array of number of clients.
        rounds_array: An array of number of communication rounds.
        strategy_combos: A list of strategy combinations.

    Returns:
        None
    """
    for strategy_combo in strategy_combos:
        for samplemodel in models:
            for num_clients in num_clients_array:
                save.save_num_clients(num_clients)
                for rounds in rounds_array:
                        # Make sure model isnt reused
                        model = samplemodel.copy()
                        #def get_strategy(model, numrounds, strategycombo, name):
                        name = "model_"+ get_dataset_name() + "_" + str(rounds) + "_" + strategy_combo.get('name') + "_" + str(num_clients) + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        strategy = get_strategy(model, rounds, strategy_combo, name)
                        save.save_parameters(name, num_clients, rounds, strategy_combo, model.getsetupvalues())
                        # model = get_model()
                        print(" starting full run with the folling parameters: ")
                        print(" num_clients " + str(num_clients))
                        print(" rounds " + str(rounds))
                        print(" model ", model.get_parameters_string())
                        print(" strategy "  + strategy_combo.get('name'))
                        print(" strategy_parameters " + str(strategy_combo))
                        full_single_run(model, num_clients, rounds, strategy, name)

if __name__ == "__main__":
    possible_num_clients = read.get_numclients()
    possible_rounds = read.get_numrounds()
    print("Starting gridsearch for FL Neural Network")
    models = get_models()
    strategy_combos = get_strategy_combos()
    full_gridsearch_run(models, possible_num_clients, possible_rounds, strategy_combos)
