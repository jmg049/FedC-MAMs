from dataclasses import dataclass

from config.config import Config


@dataclass(kw_only=True)
class FederatedIncongreuntConfig(Config):
    pass
