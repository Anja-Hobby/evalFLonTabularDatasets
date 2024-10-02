from omegaconf import DictConfig
import save
import plot
FILENAME = "server"

def get_on_fit_config(config: DictConfig):
    """
    Return a function to configure the client's fit.

    Args:
        config (DictConfig): The configuration dictionary.

    Returns:
        function: A function that takes the server round as input and returns a fit configuration dictionary.
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

    Parameters:
    - model: The global model to be evaluated.
    - serverdata: The data used for evaluation.
    - max_rounds: The maximum number of rounds for evaluation.
    - name: The name of the model.

    Returns:
    - evaluate_fn: The evaluation function.
    """
    metrics_over_time = []

    def evaluate_fn(server_round: int, coefs, config):
        """
        Evaluate the global model.

        Parameters:
        - server_round: The current round of evaluation.
        - coefs: The coefficients of the model.
        - config: The configuration of the model.

        Returns:
        - loss: The loss value of the evaluation.
        - metrics: The metrics of the evaluation.
        """
        model.set_parameters(coefs)
        loss, amountofdata, metrics = model.evaluate(serverdata[0], serverdata[1])
        metrics_over_time.append(metrics)
        if server_round == max_rounds:
            #model.plot(serverdata[0], serverdata[1])
            #plot.plot_graphs(metrics_over_time, name)
            save.final_results(metrics_over_time, name)
            save.save_model_params(model, name)
            
        return loss, metrics

    return evaluate_fn