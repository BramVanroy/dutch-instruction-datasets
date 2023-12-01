from typing import Any


def dict_to_tuple(d: dict) -> tuple[tuple[Any, Any], ...]:
    """
    Convert a dict to a tuple of sorted items. Useful when working with functions (like lru_wrapped ones) that don't
    accept unhashable ttypes as arguments.
    :param d: dict to convert
    :return: tuplified dict
    """
    return tuple(sorted(d.items()))
