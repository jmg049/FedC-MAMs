from torch.nn import init
from torch.nn import Conv2d, Linear, BatchNorm2d
from models.networks import LSTMEncoder, TextCNN


def kaiming_init(module):
    if isinstance(module, (Conv2d, Linear)):
        init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)


def resolve_encoder(_type: str):
    match _type.lower():
        case "lstmencoder":
            return LSTMEncoder
        case "textcnn":
            return TextCNN
        case _:
            raise ValueError(f"Unknown encoder type: {_type}")
