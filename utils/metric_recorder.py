from functools import partial
import json
import os
from typing import Dict, Any, List, Callable, Union
import yaml
import importlib
import torch
from logging import getLogger

from collections import OrderedDict

logger = getLogger("cmams")


class MetricRecorder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics: OrderedDict[str, Callable] = self._load_metrics()
        self.results: OrderedDict[str, List[float]] = {
            metric: [] for metric in self.metrics
        }

    def _load_metrics(self) -> OrderedDict[str, Callable]:
        metrics = OrderedDict()
        for metric_name, metric_info in self.config.items():
            module_name, func_name = metric_info["function"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            metric_func = getattr(module, func_name)
            # Use .get() to safely retrieve kwargs, defaulting to an empty dict if not present
            metrics_kwargs = metric_info.get("kwargs", {})
            metrics[metric_name] = partial(metric_func, **metrics_kwargs)

        return metrics

    def clone(self):
        return MetricRecorder(self.config)

    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        skip_metrics: List[str] = None,
    ) -> Dict[str, float]:
        if skip_metrics is None:
            skip_metrics = []
        if isinstance(predictions, torch.Tensor):
            if predictions.device != "cpu":
                predictions = predictions.detach().cpu()

            predictions = predictions.numpy()

        if isinstance(targets, torch.Tensor):
            if targets.device != "cpu":
                targets = targets.detach().cpu()

            targets = targets.numpy()

        results = OrderedDict()
        for metric_name, metric_func in self.metrics.items():
            if metric_name in skip_metrics:
                continue
            results[metric_name] = metric_func(targets, predictions)
        return results

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        results = self.calculate_metrics(predictions, targets)
        for metric_name, value in results.items():
            if isinstance(value, dict):
                for m, v in value.items():
                    self.results[f"{metric_name}_{m}"].append(v)
            else:
                self.results[metric_name].append(value)

    def update_from_dict(
        self,
        results: Dict[str, float],
    ):
        for metric_name in results.keys():
            if metric_name not in self.results:
                self.results[metric_name] = []
            self.results[metric_name].append(results[metric_name])

    def get(self, metric_name: str, default=None) -> Union[List[Any] | Any]:
        m = self.results.get(metric_name, default)
        if m is not None and len(m) == 1:
            return m[0]
        return m

    def get_average_metrics(self, save_to: str = None, epoch=-1) -> Dict[str, float]:
        exclude_prefixes = ["ConfusionMatrix", "classification_report"]

        m = {
            k: v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v
            for k, v in self.results.items()
            if not any(k.startswith(prefix) for prefix in exclude_prefixes)
        }

        results = OrderedDict()
        for metric, values in m.items():
            try:
                if isinstance(values, (list, tuple)) and len(values) > 0:
                    results[metric] = sum(values) / len(values)
                elif isinstance(values, (int, float)):
                    results[metric] = values
                else:
                    logger.warning(
                        f"Skipping metric '{metric}': unexpected type {type(values)}"
                    )
            except Exception as e:
                logger.exception(f"Error processing metric '{metric}': {str(e)}")
                raise Exception(f"Error processing metric '{metric}': {str(e)}")

        # save each metric to a file: epoch.json
        if save_to is not None:
            os.makedirs(save_to, exist_ok=True)
            results = {k: float(v) for k, v in results.items()}
            json_str = json.dumps(results, indent=4)
            with open(os.path.join(save_to, f"{epoch}.json"), "w") as f:
                f.write(json_str)
        return results

    def reset(self):
        for metric in self.results:
            self.results[metric] = []

    def __str__(self) -> str:
        metrics_info = []
        for metric_name, metric_func in self.metrics.items():
            func_name = metric_func.func.__name__  # Get the original function name
            module_name = metric_func.func.__module__  # Get the module name
            kwargs = metric_func.keywords  # Get the kwargs

            kwargs_str = (
                ", ".join(f"{k}={v}" for k, v in kwargs.items())
                if kwargs
                else "No additional arguments"
            )

            metric_info = f"  {metric_name}:\n    Function: {module_name}.{func_name}\n    Arguments: {kwargs_str}"
            metrics_info.append(metric_info)

        metrics_str = "\n".join(metrics_info)

        current_results = self.get_average_metrics()
        results_str = "\n".join(
            f"  {metric}: {value:.4f}" for metric, value in current_results.items()
        )

        return (
            f"MetricRecorder:\n"
            f"Configured Metrics:\n"
            f"{metrics_str}\n"
            f"Current Average Results:\n"
            f"{results_str}"
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MetricRecorder":
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)
