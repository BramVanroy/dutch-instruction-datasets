from dataclasses import dataclass
from typing import Any

from openai.types.chat import ChatCompletion
from transformers import Conversation


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
    Build a single message dictionary with role and content keys
    """
    return {
        "role": role,
        "content": content,
    }


@dataclass
class Response:
    """
    Class for storing the results of a query.
    """

    job_idx: int
    messages: list[dict[str, str]]
    result: str | ChatCompletion | None | Conversation = None
    text_response: str | None = None
    error: Exception | None = None

    def __hash__(self):
        return hash((self.job_idx, str(self.text_response)))
