from typing import Any


def dict_to_tuple(d: dict) -> tuple[tuple[Any, Any], ...]:
    """
    Convert a dict to a tuple of sorted items. Useful when working with functions (like lru_wrapped ones) that don't
    accept unhashable ttypes as arguments.
    :param d: dict to convert
    :return: tuplified dict
    """
    return tuple(sorted(d.items()))


def build_message(role: str, content: str) -> dict[str, str]:
    """
    Build a single message dictionary for the API
    """
    return {
        "role": role,
        "content": content,
    }
