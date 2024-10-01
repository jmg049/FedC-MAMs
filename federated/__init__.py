from dataclasses import dataclass, field
import json
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

    def as_json(self):
        client_info_str = json.dumps(self.client_info, indent=4)
        metrics_str = json.dumps(self.metrics, indent=4)
        metadata_str = json.dumps(self.metadata, indent=4)

        return {
            "client_info": client_info_str,
            "model_params": sum(p.numel() for p in self.model_params.values()),
            "metrics": metrics_str,
            "metadata": metadata_str,
        }
