import logging
import os
from typing import Dict

import yaml


class ConfLoader:
    """Load and process a yaml configuration file."""

    def __init__(self, path):
        self.path = path
        self.root = os.path.dirname(path)

    def load(self) -> Dict[str, any]:
        with open(self.path, "r") as f:
            conf = yaml.safe_load(f)

        self._to_absolute_paths(self.root, conf)
        return conf

    def _to_absolute_paths(self, root, conf):
        for key in conf.keys():
            if key == "path" or key.endswith("_path"):
                conf[key] = self._to_absolute(conf[key])
                if not os.path.exists(conf[key]):
                    logging.error(
                        "[ConfLoader] Path '%s' for %s does not exist.", conf[key], key
                    )
            if type(conf[key]) is dict:
                self._to_absolute_paths(root, conf[key])

    def _to_absolute(self, path):
        return os.path.abspath(os.path.join(self.root, path))
