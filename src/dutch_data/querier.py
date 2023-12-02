from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from time import sleep
from typing import Any, Generator

from dutch_data.credentials import Credentials
from dutch_data.utils import dict_to_tuple
from openai import AzureOpenAI, OpenAIError, RateLimitError
from openai._streaming import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


# A tuple with two tuples, one with content one with role, e.g.
# (("content", "msg job 1"), ("role", "user"))
MESSAGE_T = tuple[tuple[str, str], tuple[str, str]]
# conversation ID and a tuple of messages
CONVERSATION_T = tuple[int, tuple[MESSAGE_T, ...]]


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

    def update_patience(self, max_retries: int, exception: Exception | None = None, messages: Any = None):
        if exception is None and messages is None:
            raise ValueError("If an exception is not passed, messages must be passed")

        max_retries -= 1
        if max_retries == 0:
            if exception is not None:
                raise exception
            else:
                raise OpenAIError(
                    f"OpenAI API unexpectedly returned empty completion multiple times for these messages:"
                    f" {messages}"
                )

        sleep(self.timeout)

        return max_retries

    @lru_cache(maxsize=1024)
    def _query_messages(
        self, messages: tuple[int, tuple[tuple[str, str], ...]], return_full_api_output: bool = False, **kwargs
    ) -> tuple[int, str | ChatCompletion | Stream[ChatCompletionChunk]]:
        """
        Query the Azure OpenAI API.
        :param messages: tuple of messages where the first item is the index of the job, and the second item is a
         tuple of tuples of key/value pairs, to be transformerd back into a dict, e.g. ("role", "user")
        :param return_full_api_output: whether to return the full API output or only the generated text
        :param kwargs: any keyword arguments to pass to the API
        :return: tuple with the job index and generated text or full API output, depending on 'return_full_api_output'
        """
        max_retries = self.max_retries
        # Transform messages back into required format.
        job_idx = messages[0]
        messages = [dict(message) for message in messages[1]]

        while max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.deployment_name, messages=messages, **kwargs
                )
            except Exception as exc:
                max_retries = self.update_patience(max_retries, exc)
                continue
            else:
                if not completion:
                    max_retries = self.update_patience(max_retries, messages=messages)
                    continue

                if not return_full_api_output:
                    # Only return generated text
                    completion_str = completion.choices[0].message.content
                    if not completion_str:
                        # We did not find the response in the first choice, but maybe we can find it in other choices
                        # If the model accidentally returned multiple choices?
                        for idx in range(1, len(completion.choices)):
                            completion_str = completion.choices[idx].message.content
                            if completion_str:
                                return job_idx, completion_str

                        # Still did not find the response, so retry
                        max_retries = self.update_patience(max_retries, messages=messages)
                        continue
                    return job_idx, completion_str
                return job_idx, completion

    def query_list_of_messages(
        self,
        list_of_messages: list[list[ChatCompletionMessageParam]] | list[tuple[int, list[ChatCompletionMessageParam]]],
        return_full_api_output: bool = False,
        return_in_order: bool = True,
        **kwargs,
    ) -> Generator[tuple[int, str | ChatCompletion | Stream[ChatCompletionChunk]], None, None]:
        """
        Query the Azure OpenAI API with a list of messages.
        :param list_of_messages: list of lists of messages to send to the API. We will add job idxs automatically
        but for more control you can also pass a list of tuples of (job_idx, messages) to specify the job idxs yourself
        :param return_full_api_output: whether to return the full API output or only the generated text
        :param return_in_order: whether to return the results in the order of the input
        :param kwargs: any keyword arguments to pass to the API
        :return: a list of tuples with job indexes as the first item and as the second the generated text or full API
         outputs, depending on 'return_full_api_output'
        """
        add_job_idx = not isinstance(list_of_messages[0][0], int)
        if add_job_idx:
            list_of_messages = tuple(
                [
                    (job_idx, tuple([dict_to_tuple(message_d) for message_d in messages]))
                    for job_idx, messages in enumerate(list_of_messages)
                ]
            )
        else:
            list_of_messages = tuple(
                [
                    (job_idx, tuple([dict_to_tuple(message_d) for message_d in messages]))
                    for job_idx, messages in list_of_messages
                ]
            )

        if self.max_workers < 2:
            for messages in list_of_messages:
                yield self._query_messages(messages, return_full_api_output, **kwargs)
        else:
            with ThreadPoolExecutor(self.max_workers) as executor:
                futures = [
                    executor.submit(self._query_messages, idx_and_messages, return_full_api_output, **kwargs)
                    for idx_and_messages in list_of_messages
                ]

                yielder = futures if return_in_order else as_completed(futures)
                for future in yielder:
                    # Should trigger an exception here if the future failed inside self.query_messages
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
