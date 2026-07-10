from omegaconf import OmegaConf
from types import SimpleNamespace

class WandbConfigSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise Exception("WandbConfigSingleton has not been initialized")
        return cls._instance

    @classmethod
    def initialize(cls, run, llm, config_override=None):
        if cls._instance is not None:
            raise Exception("WandbConfigSingleton has already been initialized")
        # Convert run.config to a standard Python dictionary unless the caller
        # has already resolved runtime-only values such as run-scoped outputs.
        config_dict = config_override if config_override is not None else dict(run.config)
        # Convert Python dictionary to DictConfig
        config = OmegaConf.create(config_dict)
        # Store as attributes in _instance
        cls._instance = SimpleNamespace(run=run, config=config, blend_config=None, llm=llm)
