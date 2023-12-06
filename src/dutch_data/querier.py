import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from time import sleep
from typing import Any, Generator

from dutch_data.credentials import Credentials
from dutch_data.utils import dict_to_tuple
from openai import AzureOpenAI, BadRequestError, OpenAIError, RateLimitError
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


@dataclass
class Response:
    """
    Class for storing the results of a query.
    """

    job_idx: int
    messages: list[dict[str, str]]
    result: str | ChatCompletion | None = None
    error: Exception | None = None

    def __hash__(self):
        return hash((self.job_idx, str(self.result)))


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
    verbose: bool = False
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

    def update_patience(self, remaining_retries: int, exception: Exception | None = None, messages: Any = None):
        """
        Update the patience for retrying API calls, either decreasing it or potentially throwing errors when patience
        has been exceeded.
        :param remaining_retries: remaining number of retries before checking
        :param exception: a potential exception that was thrown by the API
        :param messages: the messages that were sent to the API
        :return: the updated remaining retries
        """
        if exception is None and messages is None:
            raise ValueError("If an exception is not passed, messages must be passed")

        remaining_retries -= 1
        if remaining_retries == 0:
            if exception is not None:
                raise OpenAIError(f"An error occurred (see above) for these messages:\n{messages}") from exception
            else:
                raise OpenAIError(
                    f"OpenAI API unexpectedly returned empty completion multiple times for these messages:"
                    f" {messages}"
                )

        if self.verbose:
            print(f"Timing out for {self.timeout}s and retrying ({remaining_retries} retries left)", file=sys.stderr,
                            flush=True)

        sleep(self.timeout)

        return remaining_retries

    @lru_cache(maxsize=1024)
    def _query_messages(
        self, messages: tuple[int, tuple[tuple[str, str], ...]], return_full_api_output: bool = False, **kwargs
    ) -> Response:
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
                # Bad requests, like malformed requests or those with filtered hate speech, sexual content, etc.
                # will throw this exception. No use to retry these.
                if isinstance(exc, BadRequestError):
                    if self.verbose:
                        print(
                            f"Bad request: {exc.message}\n\nCheck your request for the following messages: {messages}",
                            file=sys.stderr,
                            flush=True
                        )
                    return Response(job_idx, messages, None, exc)
                elif isinstance(exc, RateLimitError):
                    if self.verbose:
                        print(
                            f"Rate limit exceeded ('{exc.message}'). We're timing out now and retrying later, but you may"
                            f" want to consider:\n  1. changing your API rate limits;\n  2. decreasing the number of"
                            f" parallel workers\n  3. increasing the timeout\n... and then restarting the script.",
                            file=sys.stderr,
                            flush=True
                        )

                if self.verbose:
                    print(
                        f"Exception in request: {exc.message}",
                        file=sys.stderr,
                        flush=True
                    )

                try:
                    max_retries = self.update_patience(max_retries, exc, messages=messages)
                except Exception as exc:
                    return Response(job_idx, messages, None, exc)
                continue
            else:
                if not completion:
                    if self.verbose:
                        print(
                            f"Unexpected empty completion for messages: {messages}",
                            file=sys.stderr,
                            flush=True
                        )
                    try:
                        max_retries = self.update_patience(max_retries, messages=messages)
                    except Exception as exc:
                        return Response(job_idx, messages, None, exc)
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
                                return Response(job_idx, messages, completion_str)

                        # Sometimes it seems that the model returns an empty completion, but the finish reason is
                        # "content_filter". In this case, we should not retry, but return the empty completion.
                        if completion.choices[0].finish_reason == "content_filter":
                            if self.verbose:
                                print(
                                    f"Content filter triggered for the following messages (so no response received):"
                                    f" {messages}",
                                    file=sys.stderr,
                                    flush=True
                                )
                            return Response(job_idx, messages, None, BadRequestError(response=completion, body=completion, request=None, message="Content filter triggered"))

                        # Still did not find the response, so retry
                        print(
                            f"Content response was empty for: {messages}",
                            file=sys.stderr,
                            flush=True
                        )
                        try:
                            max_retries = self.update_patience(max_retries, messages=messages)
                        except Exception as exc:
                            return Response(job_idx, messages, None, exc)
                        continue
                    return Response(job_idx, messages, completion_str)
                return Response(job_idx, messages, completion)

    def query_list_of_messages(
        self,
        list_of_messages: list[list[ChatCompletionMessageParam]] | list[tuple[int, list[ChatCompletionMessageParam]]],
        return_full_api_output: bool = False,
        return_in_order: bool = True,
        **kwargs,
    ) -> Generator[Response, None, None]:
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
