import base64
import io
import torch


def serialize_parameters(parameters: dict[str, torch.Tensor]) -> dict[str, str]:
    """
    Serializes model parameters for federated learning.

    Args:
    parameters (dict[str, torch.Tensor]): A dictionary of model parameters.

    Returns:
    dict[str, str]: A dictionary with the same keys, but values serialized to base64 strings.
    """
    serialized_params = {}
    for name, param in parameters.items():
        buffer = io.BytesIO()
        torch.save(param.data, buffer)
        serialized_params[name] = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return serialized_params


def deserialize_parameters(
    serialized_params: dict[str, str],
) -> dict[str, torch.Tensor]:
    """
    Deserializes model parameters for federated learning.

    Args:
    serialized_params (dict[str, str]): A dictionary of serialized model parameters.

    Returns:
    dict[str, torch.Tensor]: A dictionary with the same keys, but values deserialized to torch.Tensor.
    """
    deserialized_params = {}
    for name, param_str in serialized_params.items():
        buffer = io.BytesIO(base64.b64decode(param_str))
        deserialized_params[name] = torch.load(buffer)
    return deserialized_params
