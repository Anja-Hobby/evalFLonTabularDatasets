import yaml

def get_dataset_loader():
    """
    Retrieves the dataset location and target from the configuration file.

    Returns:
        list: A list containing the dataset location and target.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Extract dataset information
    dataset_location = config['dataset']['location']
    dataset_target = config['dataset']['target']

    # Instantiate currentData
    currentData = [dataset_location, dataset_target]
    return currentData

def get_numrounds():
    """
    Retrieves the number of rounds from the configuration file.

    Returns:
        int: The number of rounds.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['num_rounds']

def get_numclients():
    """
    Retrieves the number of clients from the configuration file.

    Returns:
        int: The number of clients.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['num_clients']

def model_dictionary():
    """
    Retrieves the model dictionary from the configuration file.

    Returns:
        dict: The model dictionary.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['model']

def use_smote():
    """
    Retrieves the 'use_smote' flag from the configuration file.

    Returns:
        bool: The value of the 'use_smote' flag.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['use_smote']

def use_undersample():
    """
    Retrieves the 'use_undersample' flag from the configuration file.

    Returns:
        bool: The value of the 'use_undersample' flag.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['use_undersample']

def get_valratio_clients():
    """
    Retrieves the validation ratio for clients from the configuration file.

    Returns:
        float: The validation ratio for clients.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['valratio_clients']

def get_modelsetupfeatures():
    """
    Retrieves the model setup features from the configuration file.

    Returns:
        list: A list of model setup features.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['modelsetupfeatures']

def get_debugging():
    """
    Retrieves the debugging flag from the configuration file.

    Returns:
        bool: The value of the debugging flag.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['debugging']

def get_strategy():
    """
    Retrieves the strategy from the configuration file.

    Returns:
        str: The strategy.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['strategy']