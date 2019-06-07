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


def write_config(content, fname):
    with fname.open('wt') as handle:
        handle.write("CONFIG = " + str(content))
        handle.close()


def retrieve_sub_folder_paths(root):
    dir_paths = []
    for subdir in filter(lambda x: os.path.isdir(x), map(lambda x: os.path.join(root, x), os.listdir(root))):
        for subsubdir in filter(lambda x: os.path.isdir(x), map(lambda x: os.path.join(subdir, x), os.listdir(subdir))):
            dir_paths.append(subsubdir)
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
