import json

from utils.model_type import ModelType
from types import SimpleNamespace

with open('config.json') as jsfile:
    config = json.load(
        jsfile, object_hook=lambda i: SimpleNamespace(**i))


def get_config():
    return config
