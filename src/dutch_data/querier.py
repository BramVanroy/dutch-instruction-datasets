from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from time import sleep
from typing import Generator

from dutch_data.credentials import Credentials
from openai import AzureOpenAI, RateLimitError
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
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    def query_messages(
        self, messages: list[ChatCompletionMessageParam], return_full_api_output: bool = False, **kwargs
    ) -> str | ChatCompletion | Stream[ChatCompletionChunk]:
        """
        Query the Azure OpenAI API.
        :param messages: list of messages to send to the API
        :param return_full_api_output: whether to return the full API output or only the generated text
        :param kwargs: any keyword arguments to pass to the API
        :return: generated text or full API output, depending on 'return_full_api_output'
        """
        max_retries = self.max_retries
        while max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.deployment_name, messages=messages, **kwargs
                )
            except RateLimitError as exc:
                max_retries -= 1
                if max_retries == 0:
                    raise exc

                sleep(self.timeout)
                continue
            else:
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
        if self.max_workers < 2:
            for messages in list_of_messages:
                yield self.query_messages(messages, return_full_api_output, **kwargs)
        else:
            with ThreadPoolExecutor(self.max_workers) as executor:
                futures = [
                    executor.submit(self.query_messages, messages, return_full_api_output, **kwargs)
                    for messages in list_of_messages
                ]

                yielder = futures if return_in_order else as_completed(futures)
                for future in yielder:
                    yield future.result()

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
