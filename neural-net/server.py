from collections import OrderedDict
from omegaconf import DictConfig
import torch
import save
import plot
import model as mod
FILENAME = "server"
def get_on_fit_config(config: DictConfig):
    """Return a function to configure the client's fit.

    Args:
        config (DictConfig): The configuration object containing the fit parameters.

    Returns:
        fit_config_fn (function): A function that configures the client's fit for a given round.

    """
    def fit_config_fn(server_round: int):
        print(f"Server: Configuring fit for round {server_round}.")
        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(model, serverdata, max_rounds: int, name):
    """
    Return a function to evaluate the global model.

    Args:
        model (torch.nn.Module): The global model to be evaluated.
        serverdata (tuple): A tuple containing the server data used for evaluation.
        max_rounds (int): The maximum number of rounds for evaluation.
        name (str): The name of the model.

    Returns:
        evaluate_fn (function): A function that evaluates the global model for a given round.

    """
    metrics_over_time = []

    def evaluate_fn(server_round: int, coefs, config):
        """
        Evaluates the model for a given server round.

        Args:
            server_round (int): The current server round.
            coefs: The coefficients used to update the model.
            config: The configuration parameters.

        Returns:
            tuple: A tuple containing the loss and a dictionary of evaluation metrics.
        """
        print(f"Server: Evaluating model for round {server_round}.")
        device = torch.device("cpu")
        params_dict = zip(model.state_dict().keys(), coefs)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy, fscore, precision, recall = mod.test(model, [serverdata[0], serverdata[1]], device)
        metrics_over_time.append({"accuracy": accuracy, "f1_score": fscore, "precision": precision, "recall": recall})
        if server_round == max_rounds:
            model.plot(serverdata[0], serverdata[1])
            # plot.plot_graphs(metrics_over_time, name)
            save.final_results(metrics_over_time, name)
            save.save_model_params(model, name)
        return loss, {"accuracy": accuracy, "f1_score": fscore, "precision": precision, "recall": recall}

    return evaluate_fn