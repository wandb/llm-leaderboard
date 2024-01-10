from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace

class WandbConfigSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise Exception("WandbConfigSingleton has not been initialized")
        return cls._instance

    @classmethod
    def initialize(cls, run, table):
        if cls._instance is not None:
            raise Exception("WandbConfigSingleton has already been initialized")
        # Convert run.config to a standard Python dictionary
        config_dict = dict(run.config)
        # Convert Python dictionary to DictConfig
        config = OmegaConf.create(config_dict)
        # Store as attributes in _instance
        cls._instance = SimpleNamespace(run=run, config=config, table=table)
