import re


def extract_conversation_from_json(maybe_invalid_jsonstr: str) -> list[dict[str, str]]:
    """
    Extracts a conversation from a JSON string that is not necessaruly valid JSON.
    :param maybe_invalid_jsonstr: JSON string that may be invalid that must contain
    '"user": "..."', '"assistant": "..."' messages to extract
    :return: list of messages, where each message is a dictionary with keys 'role' and 'content'
    """
    matches = re.finditer(
        r"\"user\":\s*\"(.*?)\",\s*\"assistant\":\s*\"(.*?)(\"\s*}?|$)",
        maybe_invalid_jsonstr,
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )

    messages = []
    for match in matches:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": match.group(1).strip(),
                },
                {
                    "role": "assistant",
                    "content": match.group(2).strip(),
                },
            ]
        )
    return messages
