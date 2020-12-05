import configparser

config = configparser.ConfigParser()
config.read('config.ini')


def get_section(section):
    return config[section]


def get_preprocess():

    section = get_section('PREPROCESS')
    section.setdefault('FORCE_CPU', 1)
    return section
