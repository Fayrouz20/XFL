from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

# Import model and utility functions from task.py
from .task import Net, get_weights, set_weights

class XFLStrategy(fl.server.strategy.Strategy):
    def _init_(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        num_layers: int = 9,  # Updated to match the number of trainable layers in the model
        initial_learning_rate: float = 0.01,  # Initial learning rate
        learning_rate_decay: float = 0.9,  # Learning rate decay factor
    ) -> None:
        """Initialize the custom federated learning strategy."""
        super()._init_()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.num_layers = num_layers
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_decay = learning_rate_decay

    def _repr_(self) -> str:
        return "XFLStrategy"

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        net = Net()
        ndarrays = get_weights(net)
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        fit_configurations = []
        for idx, client in enumerate(clients):
            layer_index = self.get_cyclic_layer(server_round, idx)
            current_lr = self.initial_learning_rate * (self.learning_rate_decay ** server_round)
            config = {
                "layer_index": layer_index,
                "lr": current_lr
            }
            fit_configurations.append((client, FitIns(parameters, config)))

        return fit_configurations

    def get_cyclic_layer(self, round_num: int, node_rank: int) -> int:
        """Generate unique layer indices for different nodes and rounds."""
        return (round_num + node_rank) % self.num_layers

    def get_size_of_updates(self, updates):
        """Calculate the size of the updates in bytes, kilobytes, and megabytes."""
        total_elements = sum(param.size for param in updates)
        # Convert the size to bytes
        total_size_bytes = total_elements * 4  # a float is 4 bytes
        # Convert the size to kilobytes
        total_size_kb = total_size_bytes / 1024
        # Convert the size to megabytes
        total_size_mb = total_size_kb / 1024
        return total_size_bytes, total_size_kb, total_size_mb

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Calculate the size of the updates
        updates = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        total_size_bytes, total_size_kb, total_size_mb = self.get_size_of_updates(updates[0])

        # Communication cost tracking
        total_data_transmitted = sum(fit_res.num_examples for _, fit_res in results)
        metrics_aggregated = {
            "communication_cost": total_data_transmitted,
            "update_size_bytes": total_size_bytes,
            "update_size_kb": total_size_kb,
            "update_size_mb": total_size_mb,
        }
        print(f"Round {server_round} communication cost: {total_data_transmitted}")
        print(f"Round {server_round} update size: {total_size_bytes} bytes, {total_size_kb:.2f} KB, {total_size_mb:.2f} MB")

        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        config = {}
        evaluate_ins = EvaluateIns(parameters, config)
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )

        # Track aggregated evaluation metrics
        accuracy_aggregated = np.mean([evaluate_res.metrics["accuracy"] for _, evaluate_res in results])
        metrics_aggregated = {"accuracy": accuracy_aggregated}
        print(f"Round {server_round} evaluation - Loss: {loss_aggregated:.4f}, Accuracy: {accuracy_aggregated:.4f}")

        return loss_aggregated, metrics_aggregated

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        loss = np.random.rand()
        accuracy = np.random.rand()
        print(f"Evaluating round {server_round}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        return loss, {"accuracy": accuracy}

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients