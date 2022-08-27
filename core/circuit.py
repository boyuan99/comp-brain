import numpy as np
import yaml


class Circuit:
    def __init__(self, **kargs):
        if 'config_file' in kargs.keys():
            with open(kargs['config_file'], 'r') as config_file:
                self.config = yaml.safe_load(config_file)
                config_file.close()

        self.name_scope = None

