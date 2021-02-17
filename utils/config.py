import json
from bunch import Bunch
import os
from utils.utils import get_project_root



def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file):
    root_path = get_project_root()
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join(root_path, "./experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join(root_path, "./experiments", config.exp_name, "checkpoint/")
    return config
