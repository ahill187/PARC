import os
import psutil
import numpy as np

MEMORY_THRESHOLD = 0.0625 # GiB
MEMORY_PRUNE_GLOBAL = 0.22 / 10**6 # GiB / edge


def get_mode(a_list):
    """Get the value which appears most often in the list (the mode).

    If multiple items are maximal, the function returns the first one encountered.

    Args:
        a_list (list): a list with values.

    Returns:
        the most frequent value in the list
    """
    return max(set(a_list), key=a_list.count)


def get_current_memory_usage(process):
    return np.round(process.memory_info().rss / (1024**3), decimals=6)


def get_available_memory():
    return psutil.virtual_memory().available / (1024**3)


def get_total_memory():
    return psutil.virtual_memory().total / (1024**3)


def get_memory_prune_global(n_edges):
    required_memory = MEMORY_PRUNE_GLOBAL * n_edges
    available_memory = get_available_memory()
    total_memory = get_total_memory()

    return {
        "required": np.round(required_memory, 3),
        "available": np.round(available_memory, 3),
        "total": np.round(total_memory, 3),
        "is_sufficient": available_memory - required_memory - MEMORY_THRESHOLD >= 0
    }
