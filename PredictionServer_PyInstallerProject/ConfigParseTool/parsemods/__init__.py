from inspect import getsourcefile
from os.path import abspath, join, dirname, exists
import os
import sys

# get the absolute path of the current working directory
current_dir = abspath(dirname(getsourcefile(lambda: 0)))

lib_paths = [current_dir]
for lib_path in lib_paths:
    if exists(lib_path) and lib_path not in sys.path:
        print(str.format("Adding {} to sys.path", abspath(lib_path)))
        sys.path.append(abspath(lib_path))


class DictObj:
    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)