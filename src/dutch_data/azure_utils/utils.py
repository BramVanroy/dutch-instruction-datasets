import re


def extract_conversation_from_string(text: str, user_id: str = "user:", assistant_id: str = "assistant:", drop_last_if_not_assistant: bool = True) -> list[dict[str, str]]:
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
            if role:
                messages.append({"role": role, "content": content.strip()})
            role = "user"
            content = re.sub(rf"^{user_id}", "", line)
        elif line.startswith(assistant_id):
            if role:
                messages.append({"role": role, "content": content.strip()})
            role = "assistant"
            content = re.sub(rf"^{assistant_id}", "", line)
        else:
            content += line

    if role:
        messages.append({"role": role, "content": content.strip()})

    if drop_last_if_not_assistant and messages[-1]["role"] != "assistant":
        messages = messages[:-1]

    return messages
