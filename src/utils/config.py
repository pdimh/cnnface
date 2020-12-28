import configparser

from utils.model_type import ModelType

config = configparser.ConfigParser(allow_no_value=False)
config.read('config.ini')


def get_section(section):
    return {k.upper(): v for k, v in config.items(section) if len(v) > 0}


def get_preprocessing(model_type):

    section = get_section('PREPROCESSING')
    section_t = get_section(model_type.upper())
    section.update(section_t)

    if model_type == ModelType.PNET:
        section.setdefault('FORCE_CPU', 1)
        section.setdefault('ADAPTER', 'WIDERFACE')

    if model_type == ModelType.RNET:
        section.setdefault('FORCE_CPU', 0)
        section.setdefault('PYRAMID_FACTOR', 0.75)
        section.setdefault('STRIDE', 3)
        section.setdefault('MIN_SCORE', 0.6)
    return section


def get_training():

    section = get_section('TRAINING')
    section.setdefault('FORCE_CPU', 0)
    section.setdefault('GPU_MEM_LIMIT', None)
    section.setdefault('BATCH_SIZE', "100")
    section.setdefault('EPOCHS', "20")
    return section
