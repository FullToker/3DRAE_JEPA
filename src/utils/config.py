import importlib
from collections.abc import Mapping


def get_obj_from_str(path: str):
    module_name, object_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), object_name)


def instantiate_from_config(config: Mapping):
    if "target" not in config:
        raise KeyError("Model config must contain a 'target' field")
    params = dict(config.get("params", {}))
    return get_obj_from_str(config["target"])(**params)
