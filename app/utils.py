from typing import Dict
from pathlib import Path
import json
import pickle
import logging

# DEBUGGING #
import tracemalloc
import linecache
import os

logger = logging.getLogger(__name__)


def load_json(file_path: Path) -> Dict:
    with open(file_path) as f:
        return json.load(f)


def load_pickle(pkl_file_path: Path, error_msg: str = "Could not load pickle!"):
    try:
        with open(pkl_file_path, 'rb') as f:
            return pickle.load(f)
    except (OSError, EOFError):
        logger.info(error_msg)
        return {}


def save_pickle(object_, pkl_file_path: Path, error_msg: str = "Could not save pickle!"):
    try:
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(object_, f, pickle.HIGHEST_PROTOCOL)
    except (OSError, EOFError):
        logger.info(error_msg)
        return {}

# DEBUGGING
# https://stackoverflow.com/a/45679009/9209146


def display_top_malloc_lines(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))