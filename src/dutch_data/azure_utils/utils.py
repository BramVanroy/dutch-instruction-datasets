from dutch_data.utils import dict_to_tuple
from openai import AzureOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


class ContentFilterException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


def add_job_idx_to_messages(
    list_of_messages: list[list[ChatCompletionMessageParam]] | list[tuple[int, list[ChatCompletionMessageParam]]]
) -> tuple[tuple[int, tuple[tuple[tuple[str, str], ...], ...]], ...]:
    """
    Add job idxs to messages and convert into tuples (which can be serialized).
    :param list_of_messages: a list of messages, where each message is a list of tuples of key/value pairs. So
     basically a list of conversations, where each conversation is a list of messages
    :return: the modified list of messages with the job idx included and converted into tuples
    """
    add_job_idx = not isinstance(list_of_messages[0][0], int)
    if add_job_idx:
        list_of_messages = tuple(
            [
                (
                    job_idx,
                    tuple(
                        [
                            dict_to_tuple(message_d) if isinstance(message_d, dict) else message_d
                            for message_d in messages
                        ]
                    ),
                )
                for job_idx, messages in enumerate(list_of_messages)
            ]
        )
    else:
        list_of_messages = tuple(
            [
                (
                    job_idx,
                    tuple(
                        [
                            dict_to_tuple(message_d) if isinstance(message_d, dict) else message_d
                            for message_d in messages
                        ]
                    ),
                )
                for job_idx, messages in list_of_messages
            ]
        )

    return list_of_messages


class AzureOpenAIDeployment(AzureOpenAI):
    """
    Extends the AzureOpenAI class with and additional parameter for the Azure deployment.
    """

    def __init__(self, *, azure_deployment: str, **kwargs):
        self.azure_deployment = azure_deployment
        super().__init__(**kwargs)
