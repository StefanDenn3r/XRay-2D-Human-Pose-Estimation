import json
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def retrieve_sub_folder_paths(root):
    dir_paths = []
    for subdir in os.listdir(root):
        for subsubdir in os.listdir(os.path.join(root, subdir)):
            dir_paths.append(os.path.join(root, subdir, subsubdir))
    return dir_paths


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()
