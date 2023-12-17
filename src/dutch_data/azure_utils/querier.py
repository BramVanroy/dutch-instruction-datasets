import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Generator

from dutch_data.azure_utils.credentials import Credentials
from dutch_data.utils import Response, dict_to_tuple
from openai import AzureOpenAI, BadRequestError
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from tenacity import RetryError, Retrying, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type


class ContentFilterException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


def _add_job_idx_to_messages(
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
    client: AzureOpenAI | None = field(default=None, init=False)

    def __post_init__(self):
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        if self.timeout < 1:
            raise ValueError("timeout must be at least 1")

        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    def __hash__(self):
        return hash((self.api_key, self.api_version, self.deployment_name, self.endpoint))

    def _query_api(self, messages: tuple[int, tuple[tuple[tuple[str, str], ...], ...]], **kwargs) -> Response:
        """
        Query the Azure OpenAI API. This is mostly intended for internal use (because it requires having an index and
        converting the messages into the required format), but can be used externally as well.
        :param messages: tuple of messages where the first item is the index of the job, and the
        second item is a tuple of tuples of key/value pairs, to be transformed back into a dict, e.g. ("role", "user")
        :param kwargs: any keyword arguments to pass to the API
        :return: Response object with results and/or potential errors
        """
        if not isinstance(messages[0], int) or not isinstance(messages[1], tuple):
            raise ValueError(
                f"Expected a batch (list/tuple) of messages. Messages must be a tuple of (job_idx, messages), but got"
                f" ({type(messages[0])}, {type(messages[1])}).  This input format is mostly intended"
                f" for internal use, but can be used externally as well, so for an easier entrypoint it is recommended"
                f" to use query_list_of_messages instead, which works on 'lists' of messages instead of singletons."
            )

        # Transform messages back into required format.
        job_idx = messages[0]
        messages = [dict(message) for message in messages[1]]
        user_only_messages = [
            {"role": "user", "content": message["content"]} for message in messages if message["role"] == "user"
        ]
        response = {
            "job_idx": job_idx,
            "messages": messages,
            "result": None,
            "text_response": None,
            "error": None,
        }

        try:
            for attempt in Retrying(
                retry=retry_if_not_exception_type(BadRequestError),
                wait=wait_random_exponential(min=1, max=self.timeout),
                stop=stop_after_attempt(self.max_retries),
            ):
                with attempt:
                    completion = self.client.chat.completions.create(
                        model=self.deployment_name, messages=messages, **kwargs
                    )
        except BadRequestError as exc:
            response["error"] = ContentFilterException(
                f"Bad request error. Your input was likely malformed or contained inappropriate requests."
                f" More details:\n{exc.message}"
            )
        except RetryError as retry_error:
            if retry_error.last_attempt.failed:
                response["error"] = retry_error.last_attempt.exception()
            else:
                response["error"] = retry_error
        except Exception as exc:
            response["error"] = exc
        else:
            choice = completion.choices[0]
            response["result"] = completion
            # Sometimes it seems that the model returns an empty completion, but the finish reason is
            # "content_filter". In this case, we should not retry, but return the empty completion.
            if choice.finish_reason == "content_filter":
                if self.verbose:
                    print(
                        f"Content filter triggered for the following user messages (system message hidden)"
                        f" (so no response received):\n{user_only_messages}",
                        file=sys.stderr,
                        flush=True,
                    )
                response["error"] = ContentFilterException("Content filter triggered as a reason for completion")
            else:
                if choice.finish_reason == "length":
                    if self.verbose:
                        print(
                            f"Length limit reached for the following messages so the response might be incomplete."
                            f" Consider increasing your 'max_tokens' parameter. Check the model page for each model's "
                            f" max token limit. https://platform.openai.com/docs/models. Messages:\n"
                            f" {messages}",
                            file=sys.stderr,
                            flush=True,
                        )
                response["text_response"] = choice.message.content

        return Response(**response)

    def query_messages(self, messages: list[ChatCompletionMessageParam], **kwargs) -> Response:
        """
        Query the Azure OpenAI API with a single conversation (list of turns (typically dictionaries)).
        :param messages: a single conversation, so a list of turns (typically dictionaries) to send to the API
        :param kwargs: any keyword arguments to pass to the API
        :return: Response object with results and/or potential errors
        """
        return next(self.query_list_of_messages([messages], **kwargs))

    def query_list_of_messages(
        self,
        list_of_messages: list[list[ChatCompletionMessageParam]] | list[tuple[int, list[ChatCompletionMessageParam]]],
        return_in_order: bool = True,
        **kwargs,
    ) -> Generator[Response, None, None]:
        """
        Query the Azure OpenAI API with a list of messages.
        :param list_of_messages: list of lists of messages to send to the API. We will add job idxs automatically
        but for more control you can also pass a list of tuples of (job_idx, messages) to specify the job idxs yourself
        :param return_in_order: whether to return the results in the order of the input
        :param kwargs: any keyword arguments to pass to the API
        :return: a generator of Response objects
        """

        if (
            not isinstance(list_of_messages, list)
            or not isinstance(list_of_messages[0], (tuple, list))
            or not isinstance(list_of_messages[0][0], (int, dict))
        ):
            example_input = [
                [{"role": "system", "content": "You are a good AI"}, {"role": "user", "content": "hi"}],
                {"role": "user", "content": "good morning"},
            ]
            pretty_example = json.dumps(example_input, indent=2)
            raise ValueError(
                f"list_of_messages is expected to be a list of conversations, where each conversation is a list of"
                f" messages. For instance,\n{pretty_example}"
                f"\nFor advanced use cases, you could also add job idxs yourself by passing a list of tuples of"
                f" (job_idx, messages) where messages is still expected to be a list of dicts (or compatible)."
            )

        list_of_messages = _add_job_idx_to_messages(list_of_messages)

        if self.max_workers < 2:
            for idx_and_messages in list_of_messages:
                yield self._query_api(idx_and_messages, **kwargs)
        else:
            with ThreadPoolExecutor(self.max_workers) as executor:
                futures = [
                    executor.submit(self._query_api, idx_and_messages, **kwargs)
                    for idx_and_messages in list_of_messages
                ]

                yielder = futures if return_in_order else as_completed(futures)
                for future in yielder:
                    # Should trigger an exception here if the future failed inside self.query_messages
                    yield future.result()

    @classmethod
    def from_credentials(
        cls,
        credentials: Credentials,
        max_retries: int = 3,
        timeout: float = 30.0,
        max_workers: int = 6,
        verbose: bool = False,
    ):
        """
        Initialize AzureQuerier from Credentials object.
        :param credentials: Credentials object
        :param max_retries: maximum number of retries for API calls
        :param timeout: timeout for API calls
        :param max_workers: maximum number of workers for multi-threaded querying
        :param verbose: whether to print more information of the API responses
        :return: Initialized AzureQuerier object
        """
        return cls(
            **asdict(credentials), max_retries=max_retries, timeout=timeout, max_workers=max_workers, verbose=verbose
        )
