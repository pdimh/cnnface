import configparser

config = configparser.ConfigParser()
config.read('config.ini')


def get_section(section):
    return config[section]


def get_preprocess():

    section = get_section('PREPROCESSING')
    section.setdefault('FORCE_CPU', "1")
    return section


def get_training():

    section = get_section('TRAINING')
    section.setdefault('FORCE_CPU', "0")
    section.setdefault('GPU_MEM_LIMIT', "")
    section.setdefault('BATCH_SIZE', "100")
    section.setdefault('EPOCHS', "20")
    return section
