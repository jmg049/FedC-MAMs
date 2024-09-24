from dataclasses import dataclass, field
from typing import Any, Dict
import torch
from pprint import pformat

from utils.io import serialize_parameters, deserialize_parameters


@dataclass
class FederatedResult:
    client_info: Dict[str, Any]
    model_params: Dict[str, torch.Tensor]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        data = {
            "Client Info": self.client_info,
            "Model Params": sum(p.numel() for p in self.model_params.values()),
            "Metrics": self.metrics,
            "Metadata": self.metadata,
        }
        return pformat(data)

    def serialize(self):
        return {
            "client_info": self.client_info,
            "model_params": serialize_parameters(self.model_params),
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]):
        return cls(
            client_info=data["client_info"],
            model_params=deserialize_parameters(data["model_params"]),
            metrics=data["metrics"],
            metadata=data.get("metadata", {}),
        )
