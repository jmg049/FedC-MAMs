from dataclasses import dataclass


from typing import Optional


@dataclass
class FederatedClientConfig:
    n_rounds: int
    epochs: int
    output_dir: str
    target_metric: str = "loss"
    do_validation_visualization: bool = False


@dataclass(kw_only=True)
class FederatedConfig:
    client_config: FederatedClientConfig
    num_clients: int
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    evaluate_fn: Optional[callable] = None
    on_fit_config_fn: Optional[callable] = None
    on_evaluate_config_fn: Optional[callable] = None
    initial_parameters: Optional[dict] = None
    fit_metrics_aggregation_fn: Optional[callable] = None
    evaluate_metrics_aggregation_fn: Optional[callable] = None

    def __post_init__(self):
        if self.min_fit_clients > self.num_clients:
            raise ValueError("min_fit_clients cannot be greater than num_clients")
        if self.min_evaluate_clients > self.num_clients:
            raise ValueError("min_evaluate_clients cannot be greater than num_clients")
        if self.min_available_clients > self.num_clients:
            raise ValueError("min_available_clients cannot be greater than num_clients")
