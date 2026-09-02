# This file is no longer needed since we're using OmegaConf directly
# All configuration is now handled through YAML files and the DictConfig object

from omegaconf import DictConfig

# This module is kept for backward compatibility but is now deprecated
# All configuration should be done through the OmegaConf system

def deprecated_function():
    """
    This module is deprecated. Use OmegaConf configuration directly.
    """
    raise NotImplementedError(
        "arguments.py is deprecated. Use OmegaConf configuration directly."
    )
