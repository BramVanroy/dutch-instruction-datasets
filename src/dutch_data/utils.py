import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, TypeVar

from openai.types.chat import ChatCompletion
from transformers import Conversation


def dict_to_tuple(d: dict, do_sort: bool = True) -> tuple[tuple[Any, Any], ...]:
    """
    Convert a dict to a tuple of sorted items. Useful when working with functions (like lru_wrapped ones) that don't
    accept unhashable types as arguments.
    :param do_sort: whether to sort the items by key
    :param d: dict to convert
    :return: tuplified dict
    """
    if do_sort:
        return tuple(sorted(d.items()))
    else:
        return tuple(d.items())


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
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None = None
    extra_args: tuple | None = None

    def __hash__(self):
        return hash((self.job_idx, str(self.text_response), self.finish_reason, self.extra_args))


T = TypeVar("T", bound=Iterable)


def batchify(list_of_items: T, batch_size: int) -> list[T]:
    """
    Batchify a list of items into a list of batches (lists) of items.
    :param list_of_items: list of items to batchify
    :param batch_size: batch size
    :return: batchified list
    """
    return [list_of_items[i : i + batch_size] for i in range(0, len(list_of_items), batch_size)]


def extract_conversation_from_string(
    text: str, user_id: str = "user:", assistant_id: str = "assistant:", drop_last_if_not_assistant: bool = True
) -> list[dict[str, str]]:
    """
    Extracts a conversation from a string. Assumes that the string is formatted as follows:
    ```
    {user_id}: [message-1]
    {assistant_id}: [response to message-1]

    {user_id}: [message-2]
    {assistant_id}: [response to message-2]
    ```

    :param text: text to extract the conversation from
    :param user_id: id description of the user. Make sure to include colons or other characters that are
    part of the identifier
    :param assistant_id: id description of the assistant. Make sure to include colons or other characters that are
    part of the identifier
    :param drop_last_if_not_assistant: whether to drop the last message if it is not an assistant message
    :return: list of messages, where each message is a dictionary with keys 'role' and 'content'
    """

    messages = []
    role = None
    content = ""

    for line in text.splitlines(keepends=True):
        if line.startswith(user_id):
            if role and (content := content.strip()):
                messages.append({"role": role, "content": content})
            role = "user"
            content = re.sub(rf"^{user_id}", "", line)
        elif line.startswith(assistant_id):
            if role and (content := content.strip()):
                messages.append({"role": role, "content": content})
            role = "assistant"
            content = re.sub(rf"^{assistant_id}", "", line)
        else:
            content += line

    if role and (content := content.strip()):
        messages.append({"role": role, "content": content})

    if messages and drop_last_if_not_assistant and messages[-1]["role"] != "assistant":
        messages = messages[:-1]

    return messages
