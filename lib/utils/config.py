import os
import yaml


class ConfigDict(dict):
    """
    Dictionary subclass enabling attribute lookup/assignment of keys/values.
    """
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        # super(AttributeDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            value = self[key]
            if isinstance(self[key], dict):
                value = ConfigDict(value)
            return value
        raise AttributeError("object has no attribute '{}'".format(key))
    

def get_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    config = ConfigDict(config)
    _, file = os.path.split(path)
    filename = file.split('.')[0]
    config.config_name = filename
    return config
