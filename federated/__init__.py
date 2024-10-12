from dataclasses import dataclass, field
import json
from typing import Any, Dict
import torch
from pprint import pformat
from modalities import Modality

from utils.io import serialize_parameters, deserialize_parameters


@dataclass
class FederatedResult:
    metrics: Dict[str, Any]
    client_info: Dict[str, Any] = field(default_factory=dict)
    model_params: Dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    available_modality: list[Modality] = None

    def __str__(self):
        data = {
            "Client Info": self.client_info,
            "Model Params": sum(p.numel() for p in self.model_params.values()),
            "Metrics": self.metrics,
            "Metadata": self.metadata,
            "Available Modalities": self.available_modality,
        }
        return pformat(data)

    def serialize(self):
        return {
            "client_info": self.client_info,
            "model_params": serialize_parameters(self.model_params),
            "metrics": self.metrics,
            "metadata": self.metadata,
            "available_modality": self.available_modality,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]):
        return cls(
            client_info=data["client_info"],
            model_params=deserialize_parameters(data["model_params"]),
            metrics=data["metrics"],
            metadata=data.get("metadata", {}),
            available_modality=data.get("available_modality", None),
        )

    def as_json(self):
        client_info_str = json.dumps(self.client_info, indent=4)
        metrics_str = json.dumps(self.metrics, indent=4)
        metadata_str = json.dumps(self.metadata, indent=4)
        available_modality_str = json.dumps(self.available_modality, indent=4)

        return {
            "client_info": client_info_str,
            "model_params": sum(p.numel() for p in self.model_params.values()),
            "metrics": metrics_str,
            "metadata": metadata_str,
            "available_modality": available_modality_str,
        }
