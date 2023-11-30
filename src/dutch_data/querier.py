from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from time import sleep
from typing import Generator

from dutch_data.credentials import Credentials
from dutch_data.utils import dict_to_tuple
from openai import AzureOpenAI, OpenAIError, RateLimitError
from openai._streaming import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


@dataclass
class AzureQuerier:
    """
    Class for querying the Azure OpenAI API.
    """

    api_key: str
    api_version: str
    deployment_name: str
    endpoint: str
    max_retries: int = 3
    timeout: float = 30.0
    max_workers: int = 6
    client: AzureOpenAI = field(default=None, init=False)

    def __post_init__(self):
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    def __hash__(self):
        return hash((self.api_key, self.api_version, self.deployment_name, self.endpoint))

    @lru_cache(maxsize=1024)
    def query_messages(
        self, messages: tuple[tuple], return_full_api_output: bool = False, **kwargs
    ) -> str | ChatCompletion | Stream[ChatCompletionChunk]:
        """
        Query the Azure OpenAI API.
        :param messages: list of messages to send to the API
        :param return_full_api_output: whether to return the full API output or only the generated text
        :param kwargs: any keyword arguments to pass to the API
        :return: generated text or full API output, depending on 'return_full_api_output'
        """
        max_retries = self.max_retries
        # Transform messages back into required format.
        messages = [dict(message) for message in messages]

        while max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.deployment_name, messages=messages, **kwargs
                )
            except OpenAIError as exc:
                max_retries -= 1
                if max_retries == 0:
                    raise exc

                sleep(self.timeout)
                continue
            else:
                if not completion:
                    max_retries -= 1
                    if max_retries == 0:
                        raise OpenAIError(
                            f"OpenAI API unexpectedly returned empty completion multiple times for these messages:"
                            f" {messages}"
                        )

                    sleep(self.timeout)
                    continue

                if not return_full_api_output:
                    # Only return generated text
                    completion = completion.choices[0].message.content

                return completion

    def query_list_of_messages(
        self,
        list_of_messages: list[list[ChatCompletionMessageParam]],
        return_full_api_output: bool = False,
        return_in_order: bool = True,
        **kwargs,
    ) -> Generator[str | ChatCompletion | Stream[ChatCompletionChunk], None, None]:
        """
        Query the Azure OpenAI API with a list of messages.
        :param list_of_messages: list of lists of messages to send to the API
        :param return_full_api_output: whether to return the full API output or only the generated text
        :param return_in_order: whether to return the results in the order of the input
        :param kwargs: any keyword arguments to pass to the API
        :return: a list of generated text or full API outputs, depending on 'return_full_api_output'
        """
        # To ensure that examples can be cached with LRU we convert the messages to tuples (hashable type).
        list_of_messages = [tuple(dict_to_tuple(message_d) for message_d in messages) for messages in list_of_messages]
        if self.max_workers < 2:
            for messages in list_of_messages:
                result = self.query_messages(messages, return_full_api_output, **kwargs)

                if result is not None:
                    yield result
                else:
                    raise ValueError("Result is unexpectedly None")
        else:
            with ThreadPoolExecutor(self.max_workers) as executor:
                futures = [
                    executor.submit(self.query_messages, messages, return_full_api_output, **kwargs)
                    for messages in list_of_messages
                ]

                yielder = futures if return_in_order else as_completed(futures)
                for msgs, future in zip(list_of_messages, yielder):
                    # Should trigger an exception here if the future failed inside self.query_messages
                    result = future.result()
                    yield result

    @classmethod
    def from_credentials(
        cls, credentials: Credentials, max_retries: int = 3, timeout: float = 30.0, max_workers: int = 6
    ):
        """
        Initialize AzureQuerier from Credentials object.
        :param credentials: Credentials object
        :param max_retries: maximum number of retries for API calls
        :param timeout: timeout for API calls
        :param max_workers: maximum number of workers for multi-threaded querying
        :return: Initialized AzureQuerier object
        """
        return cls(**asdict(credentials), max_retries=max_retries, timeout=timeout, max_workers=max_workers)
