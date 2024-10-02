import yaml

def get_dataset_loader():
    """
    Retrieves the dataset location and target variable from the configuration file.

    Returns:
        list: A list containing the dataset location and target variable.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Extract dataset information
    dataset_location = config['dataset']['location']
    dataset_target = config['dataset']['target']

    # Instantiate currentData
    currentData = [dataset_location, dataset_target]
    return currentData

def get_model_name():
    """
    Retrieves the model name from the configuration file.

    Returns:
        str: The name of the model.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['model']

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

def get_valratio_clients():
    """
    Retrieves the validation ratio for clients from the configuration file.

    Returns:
        float: The validation ratio for clients.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['valratio_clients']

def use_smote():
    """
    Retrieves the value indicating whether to use SMOTE from the configuration file.

    Returns:
        bool: True if SMOTE should be used, False otherwise.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['use_smote']

def model_dictionary():
    """
    Retrieves the model dictionary from the configuration file.

    Returns:
        dict: The model dictionary.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['model']

def get_debugging():
    """
    Retrieves the debugging flag from the configuration file.

    Returns:
        bool: True if debugging is enabled, False otherwise.
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

def get_strategies():
    """
    Retrieves the strategies from the configuration file.

    Returns:
        list: A list of strategies.
    """
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config['strategies']