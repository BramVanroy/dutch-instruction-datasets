import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import cycle
from os import PathLike
from pathlib import Path
from typing import Generator, Iterator

from dutch_data.api_utils.credentials import OpenAiCredentials
from dutch_data.api_utils.utils import ContentFilterException, add_job_idx_to_messages
from dutch_data.text_generator.base import TextGenerator
from dutch_data.utils import Response
from openai import BadRequestError
from openai import OpenAI as OpenAiClient
from tenacity import RetryError, Retrying, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_not_exception_type


@dataclass
class OpenAiTextGenerator(TextGenerator):
    """
    Text generator for the OpenAI API.
    """

    clients: OpenAiClient | list[OpenAiClient]
    models: str | list[str]
    max_workers: int = 6
    verbose: bool = False
    cyclical_clients: Iterator[OpenAiClient] | None = field(init=False)
    cyclical_models: Iterator[str] | None = field(init=False)

    def __post_init__(self):
        if isinstance(self.clients, OpenAiClient):
            self.clients = [self.clients]
        if isinstance(self.models, str):
            self.models = [self.models]
        self.cyclical_clients = cycle(self.clients)
        self.cyclical_models = cycle(self.models)

    @property
    def active_client(self) -> OpenAiClient:
        return next(self.cyclical_clients)

    @property
    def active_model(self) -> str:
        return next(self.cyclical_models)

    def __hash__(self):
        return hash(
            tuple([(client.api_key, client.organization, model) for model, client in zip(self.models, self.clients)])
        )

    def _query_api(
        self, messages: tuple[int, tuple[tuple[tuple[str, str], ...], ...]], *args, json_mode: bool = False, **kwargs
    ) -> Response:
        """
        Query the OpenAI API. This is mostly intended for internal use (because it requires having an index and
        converting the messages into the required format), but can be used externally as well.
        :param messages: tuple of messages where the first item is the index of the job, and the
        second item is a tuple of tuples of key/value pairs, to be transformed back into a dict, e.g. ("role", "user")
        :param json_mode: whether to return the response in json mode or text mode. If JSON mode, the response will
        be serialized as a string with json.dumps.
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

        # Get the active client (changes on every call to 'self.active_client') so we assign it to a variable to avoid
        # it changing during the execution of this function.
        client = self.active_client
        model = self.active_model
        # Transform messages back into required format.
        job_idx = messages[0]
        messages = [dict(message) for message in messages[1]]

        response = {
            "job_idx": job_idx,
            "messages": messages,
            "result": None,
            "text_response": None,
            "error": None,
            "finish_reason": None,
            "extra_args": args if args else None,
        }

        try:
            for attempt in Retrying(
                retry=retry_if_not_exception_type(BadRequestError),
                wait=wait_random_exponential(min=1, max=client.timeout),
                stop=stop_after_attempt(client.max_retries),
            ):
                with attempt:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        response_format={"type": "json_object" if json_mode else "text"},
                        **kwargs,
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
            response["finish_reason"] = choice.finish_reason

            if choice.finish_reason == "content_filter":
                if self.verbose:
                    print(
                        f"Content filter triggered for the following messages (system message hidden)"
                        f" (so no response received):\n{messages}",
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

    def query_messages(
        self,
        messages: list[dict[str, str]] | tuple[int, list[dict[str, str]]],
        *args,
        json_mode: bool = False,
        **kwargs,
    ) -> Response:
        """
        Query the OpenAI API with a single conversation (list of turns (typically dictionaries)).
        :param messages: messages to query the model with. A list of dicts where each dict must have a "role" and
        "content" keys
        :param args: any extra arguments to send, e.g. job hash identifiers, which will be include in the Response's
        extra_args attribute
        :param json_mode: whether to return the response in json mode or text mode. If JSON mode, the response will
        be serialized as a string with json.dumps.
        :param kwargs: any keyword arguments to pass to the API
        :return: Response object with results and/or potential errors
        """
        args = [[arg] for arg in args]
        return next(self.batch_query_messages([messages], *args, json_mode=json_mode, **kwargs))

    def batch_query_messages(
        self,
        list_of_messages: list[list[dict[str, str]]] | list[tuple[int, list[dict[str, str]]], ...],
        *args,
        json_mode: bool = False,
        return_in_order: bool = True,
        **kwargs,
    ) -> Generator[Response, None, None]:
        """
        Query the OpenAI API with a list of messages.
        :param list_of_messages: list of lists of messages to send to the API. We will add job idxs automatically
        but for more control you can also pass a list of tuples of (job_idx, messages) to specify the job idxs yourself
        :param args: any extra arguments to send, e.g. job hash identifiers, which will be include in the Response's
        extra_args attribute. Must be a list of lists where the sublists are of the same length as the number of items
        in the batch.
        :param json_mode: whether to return the response in json mode or text mode. If JSON mode, the response will
        be serialized as a string with json.dumps.
        :param return_in_order: whether to return the results in the order of the input
        :param kwargs: any keyword arguments to pass to the API
        :return: a generator of Response objects
        """
        self._verify_list_of_messages(list_of_messages)
        self._verify_extra_args(args, list_of_messages)

        list_of_messages = add_job_idx_to_messages(list_of_messages)

        if self.max_workers < 2:
            for item_idx, idx_and_messages in enumerate(list_of_messages):
                item_args = [arg[item_idx] for arg in args]
                yield self._query_api(idx_and_messages, *item_args, json_mode=json_mode, **kwargs)
        else:
            with ThreadPoolExecutor(self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self._query_api,
                        idx_and_messages,
                        *[arg[item_idx] for arg in args],
                        json_mode=json_mode,
                        **kwargs,
                    )
                    for item_idx, idx_and_messages in enumerate(list_of_messages)
                ]

                yielder = futures if return_in_order else as_completed(futures)
                for future in yielder:
                    yield future.result()

    @classmethod
    def from_credentials(
        cls,
        credentials: OpenAiCredentials,
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
        api_key = credentials.api_key
        client = OpenAiClient(
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
        )
        return cls(client, models=credentials.model, max_workers=max_workers, verbose=verbose)

    @classmethod
    def from_json(
        cls,
        credentials_file: str | PathLike,
        credentials_profiles: list[str] | str | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        max_workers: int = 1,
        verbose: bool = False,
    ):
        """
        Load credentials from a JSON file to initialize a Querier from multiple profiles
        :param credentials_file: JSON file containing credentials
        :param credentials_profiles: which credential profiles (keys) to use from the credentials file. If None, will
        use all profiles
        :param max_retries: maximum number of retries for API calls
        :param timeout: timeout for API calls
        :param max_workers: maximum number of workers for multi-threaded querying
        :param verbose: whether to print more information of the API responses
        """
        pfcredentials_file = Path(credentials_file).resolve()

        if not pfcredentials_file.exists():
            raise FileNotFoundError(f"Credentials file {pfcredentials_file} does not exist.")

        credentials = json.loads(pfcredentials_file.read_text(encoding="utf-8"))
        credentials_profiles = credentials_profiles or list(credentials.keys())

        if isinstance(credentials_profiles, str):
            credentials_profiles = [credentials_profiles]

        if any(profile not in credentials for profile in credentials_profiles):
            raise KeyError(
                f"Not all profiles ({credentials_profiles}) are present in the credentials file."
                f" Available profiles: {list(credentials.keys())}"
            )

        clients = [
            OpenAiClient(
                api_key=credentials[profile]["api_key"],
                max_retries=max_retries,
                timeout=timeout,
            )
            for profile in credentials_profiles
        ]
        models = [credentials[profile]["model"] for profile in credentials_profiles]

        return cls(clients, models=models, max_workers=max_workers, verbose=verbose)
