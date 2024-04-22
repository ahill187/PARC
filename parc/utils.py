def get_mode(a_list):
    """Get the value which appears most often in the list (the mode).

    If multiple items are maximal, the function returns the first one encountered.

    Args:
        a_list (list): a list with values.

    Returns:
        the most frequent value in the list
    """
    return max(set(a_list), key=a_list.count)
