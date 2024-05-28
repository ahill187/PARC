import psutil
import numpy as np
import pandas as pd
import functools
from parc.logger import get_logger

logger = get_logger(__name__)

MEMORY_THRESHOLD = 0.0625  # GiB
MEMORY_PRUNE_GLOBAL = 0.22 / 10**6  # GiB / edge


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


def show_virtual_memory():
    virtual_memory = psutil.virtual_memory()
    virtual_memory_df = {
        "total (GiB)": np.round(virtual_memory.total / (1024**3), 6),
        "available (GiB)": np.round(virtual_memory.available / (1024**3), 6),
        "used (GiB)": np.round(virtual_memory.used / (1024**3), 6),
        "free (GiB)": np.round(virtual_memory.free / (1024**3), 6),
    }
    logger.info(f"Memory stats: {virtual_memory_df}")


def get_function_memory(n_items, memory_per_item):
    required_memory = memory_per_item * n_items
    available_memory = get_available_memory()
    total_memory = get_total_memory()

    return {
        "required": np.round(required_memory, 3),
        "available": np.round(available_memory, 3),
        "usage": np.round(total_memory - available_memory, 3),
        "total": np.round(total_memory, 3),
        "is_sufficient": available_memory - required_memory - MEMORY_THRESHOLD >= 0
    }


def check_memory(min_memory=2.0, items_kwarg=None, items_factor_kwarg="", memory_per_item=None):
    """A decorator to check the current memory usage.

    Args:
        min_memory (float): (``default=2.0``) The minimum memory required for the function, in GiB.
        items_kwarg (str): (optional) If provided, the decorator will check the function's
            ``**kwargs`` for an argument with the key provided as ``items_kwarg``. The value of the
            argument value should be either a list, a Numpy array, or a Pandas dataframe, the
            length of which is used to determine how much memory the function needs. In essence,
            this argument contains the rate-limiting data.
        memory_per_item (float): (optional) In conjunction with ``items_kwarg``, this argument
            is used to calculate the memory required for the function. So if the ``items_kwarg``
            corresponds to a list of N samples, the ``memory_per_item`` is the memory required
            per item, in GiB.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            do_check_function_memory = False
            if items_kwarg is not None and memory_per_item is not None:
                items = kwargs.get(items_kwarg, None)
                if isinstance(items, np.ndarray) or isinstance(items, pd.core.frame.DataFrame):
                    n_items = items.shape[0]
                    do_check_function_memory = True
                elif (isinstance(items, list)
                      or (hasattr(items, "__len__") and callable(items.__len__))
                ):
                    n_items = len(items)
                    do_check_function_memory = True
                elif items is None:
                    logger.warning(
                        f"Could not find item {items_kwarg} in arguments for function {func}."
                    )
                else:
                    logger.warning(
                        f"{items_kwarg} is of type {type(items_kwarg)}; must be a array-like."
                    )

            if do_check_function_memory:
                if items_factor_kwarg != "":
                    items_factor = kwargs.get(items_factor_kwarg, 1)
                    if not isinstance(items_factor, int):
                        items_factor = 1
                else:
                    items_factor = 1
                function_memory = get_function_memory(n_items * items_factor, memory_per_item)
                current_usage = function_memory["usage"]

                if not function_memory["is_sufficient"]:
                    raise MemoryError(
                        f"Not enough memory to call {func.__name__} function.\n"
                        f"available memory: {function_memory['available']} GiB\n"
                        f"required memory: {function_memory['required']} GiB\n"
                        f"You can either:\n"
                        f"a) free up memory on your computer by closing other processes;\n"
                        f"current usage: {current_usage} GiB out of {function_memory['total']}"
                        "GiB on your computer, or\n"
                        f"b) reduce the number of items in your data, which is currently "
                        f"set to {n_items} {items_kwarg} x {items_factor} {items_factor_kwarg}."
                    )
            else:
                available_memory = get_available_memory()
                if available_memory < min_memory:
                    raise MemoryError(
                        f"{available_memory} GiB left out of {get_total_memory()} GiB, "
                        "not enough memory!"
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator
