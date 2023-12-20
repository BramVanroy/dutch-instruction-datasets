from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Generator, Literal

import requests
import torch
from dutch_data.azure_utils import AzureQuerier, Credentials
from dutch_data.utils import Response
from transformers import Conversation, Pipeline, pipeline


@dataclass
class TextGenerator(ABC):
    @abstractmethod
    def query_messages(self, messages, **kwargs) -> Response:
        raise NotImplementedError

    @abstractmethod
    def batch_query_messages(self, list_of_messages, **kwargs) -> Generator[Response, None, None]:
        raise NotImplementedError


@dataclass
class HFTextGenerator(TextGenerator):
    """
    Text generator that uses locally loaded conversational pipelines.
    :param model_name: name of the model to use
    :param device_map: device map to use for the pipeline
    :param load_in_8bit: whether to load the model in 8bit precision to save memory
    :param load_in_4bit: whether to load the model in 4bit precision to save memory
    :param torch_dtype: data type to use for the model, e.g. 'float16' or 'auto'
    :param chat_template: Jinja template to use for the chat if you want to overwrite the default template that comes
    with a given model. See https://huggingface.co/transformers/main_classes/pipelines.html#transformers.ConversationalPipeline
    """

    model_name: str
    device_map: dict[str, str | int | torch.device] | str | int | torch.device = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: torch.dtype | Literal["auto"] | str | None = None
    chat_template: str | None = None
    pipe: Pipeline = field(default=None, init=False)

    def __post_init__(self):
        if self.torch_dtype is not None and self.torch_dtype != "auto":
            self.torch_dtype = getattr(torch, self.torch_dtype)

        model_kwargs = {
            "device_map": self.device_map,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "torch_dtype": self.torch_dtype,
        }
        self.pipe = pipeline(
            "conversational",
            model=self.model_name,
            model_kwargs=model_kwargs,
        )

        if self.chat_template is not None:
            self.pipe.tokenizer.chat_template = self.chat_template

    def query_messages(
        self,
        messages: list[dict[str, str]] | tuple[int, list[dict[str, str]]],
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        **kwargs,
    ) -> Response:
        """
        Query the model with a single sample of messages.
        :param messages: messages to query the model with. A list of dicts where each dict must have a "role" and
        "content" keys
        :param max_new_tokens: max new tokens to generate
        :param do_sample: whether to use sampling or not
        :param temperature: sampling temperature
        :param top_k: k for top-k sampling
        :param top_p: p for top-p sampling
        :param kwargs: additional kwargs to pass to the pipeline call
        :return: generated assistant Response
        """

        if isinstance(messages[0], int):
            job_idx = messages[0]
            messages = messages[1]
        else:
            job_idx = 0

        response = {
            "job_idx": job_idx,
            "messages": messages,
            "result": None,
            "text_response": None,
            "error": None,
        }
        try:
            conversation = self.pipe(
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            )
            response["result"] = conversation
            response["text_response"] = conversation.messages[-1]["content"]
        except Exception as exc:
            response["error"] = exc

        response = Response(**response)

        return response

    def batch_query_messages(
        self,
        list_of_messages: list[list[dict[str, str]]] | list[tuple[int, list[dict[str, str]]], ...],
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        batch_size: int = 1,
        **kwargs,
    ) -> Generator[Response, None, None]:
        """
        Query the model with the given messages.
        :param list_of_messages: messages to query the model with. A list of lists of dicts where each dict must have
         a "role" and "content" keys
        :param max_new_tokens: max new tokens to generate
        :param do_sample: whether to use sampling or not
        :param temperature: sampling temperature
        :param top_k: k for top-k sampling
        :param top_p: p for top-p sampling
        :param batch_size: batch size for the pipeline
        :param kwargs: additional kwargs to pass to the pipeline call
        :return: generator of Responses
        """
        # Interestingly, the pipeline will only yield a generator if the input was a generator or a Dataset
        # and they HAVE to be a Conversation

        if isinstance(list_of_messages[0][0], int):
            job_idxs = [item[0] for item in list_of_messages]
            list_of_messages = [item[1] for item in list_of_messages]
        else:
            job_idxs = list(range(len(list_of_messages)))

        generator_of_msgs = (Conversation(msgs) for msgs in list_of_messages)
        for item_idx, (job_idx, conversation) in enumerate(
            zip(
                job_idxs,
                self.pipe(
                    generator_of_msgs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    batch_size=batch_size,
                    **kwargs,
                ),
            )
        ):
            messages = list_of_messages[item_idx]
            yield Response(
                job_idx=job_idx,
                messages=messages,
                result=conversation,
                text_response=conversation.messages[-1]["content"],
            )


@dataclass
class AzureTextGenerator(TextGenerator):
    """
    Text generator for the Azure API.
    :param querier: AzureQuerier object to use for querying the API
    """

    querier: AzureQuerier

    def query_messages(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> Response:
        """
        Query the model with the given messages.
        :param messages: messages to query the model with. A list of dicts where each dict must have a "role" and
        "content" keys in the case of single querying, or a list of such lists in the case of batch querying.
        :param kwargs: any keyword arguments to pass to the API
        :return: generated assistant R
        """
        response = self.querier.query_messages(messages, **kwargs)

        return response

    def batch_query_messages(
        self,
        list_of_messages: list[list[dict[str, str]]],
        return_in_order: bool = False,
        **kwargs,
    ) -> Generator[Response, None, None]:
        """
        Query the model with the given batch of messages.
        :param list_of_messages: messages to query the model with. A list of lists of dicts where each dict must have
         a "role" and "content" keys
        :param return_in_order: whether to return the results in the order of the input
        :param kwargs: any keyword arguments to pass to the API
        :return: generator of Responses
        """
        yield from self.querier.query_list_of_messages(list_of_messages, return_in_order=return_in_order, **kwargs)

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
        querier = AzureQuerier.from_credentials(
            credentials, max_retries=max_retries, timeout=timeout, max_workers=max_workers, verbose=verbose
        )
        return cls(querier)

    @classmethod
    def from_json(
        cls,
        credentials_file: str | PathLike,
        credentials_profiles: list[str] | str | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        max_workers: int = 6,
        verbose: bool = False,
    ):
        """
        Initialize TextGenerator with QzureQuerier(s) from JSON file.
        :param credentials_file: JSON file containing credentials
        :param credentials_profiles: which credential profile(s) (keys) to use from the credentials file. If None, will
        use all profiles
        :param max_retries: maximum number of retries for API calls
        :param timeout: timeout for API calls
        :param max_workers: maximum number of workers for multi-threaded querying
        :param verbose: whether to print more information of the API responses
        """
        querier = AzureQuerier.from_json(
            credentials_file,
            credentials_profiles,
            max_retries=max_retries,
            timeout=timeout,
            max_workers=max_workers,
            verbose=verbose,
        )
        return cls(querier)


@dataclass
class VLLMTextGenerator(TextGenerator):
    """
    Text generator for the VLLM API.
    :param model_name: name of the model to use. Must match the model that is loaded on the VLLM server.
    :param endpoint: endpoint of the API. Note that this must be compatible with Chat Completion, as described here:
    https://docs.vllm.ai/en/latest/getting_started/quickstart.html#using-openai-completions-api-with-vllm. So this
    will likely be a URL like 'http://localhost:8000/v1/chat/completions'
    :param health_endpoint: optional health endpoint to check if the API is healthy
    """

    model_name: str
    endpoint: str
    health_endpoint: str | None = None

    @property
    def health(self):
        if not self.health_endpoint:
            return None

        response = requests.get(self.health_endpoint)
        return response.ok

    def query_messages(
        self,
        messages: list[dict[str, str]] | tuple[int, list[dict[str, str]]],
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        **kwargs,
    ) -> Response:
        """
        Query the VLLM API with a single sample of messages.
        :param messages: messages to query the model with. A list of dicts where each dict must have a "role" and
        "content" keys
        :param max_new_tokens: max new tokens to generate
        :param do_sample: whether to use sampling or not
        :param temperature: sampling temperature
        :param top_k: k for top-k sampling
        :param top_p: p for top-p sampling
        :param kwargs: additional kwargs to pass to the pipeline call
        :return: generated assistant Response
        """

        if isinstance(messages[0], int):
            job_idx = messages[0]
            messages = messages[1]
        else:
            job_idx = 0

        response = {
            "job_idx": job_idx,
            "messages": messages,
            "result": None,
            "text_response": None,
            "error": None,
            **self._submit_request(
                messages=messages,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            ),
        }
        response = Response(**response)

        return response

    def batch_query_messages(
        self,
        list_of_messages: list[list[dict[str, str]]] | list[tuple[int, list[dict[str, str]]], ...],
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        **kwargs,
    ) -> Generator[Response, None, None]:
        """
        Query the VLLM API with the given messages.
        :param list_of_messages: messages to query the model with. A list of lists of dicts where each dict must have
         a "role" and "content" keys
        :param max_new_tokens: max new tokens to generate
        :param do_sample: whether to use sampling or not
        :param temperature: sampling temperature
        :param top_k: k for top-k sampling
        :param top_p: p for top-p sampling
        :param kwargs: additional kwargs to pass to the API call
        :return: generator of Responses
        """
        if isinstance(list_of_messages[0][0], int):
            job_idxs = [item[0] for item in list_of_messages]
            list_of_messages = [item[1] for item in list_of_messages]
        else:
            job_idxs = list(range(len(list_of_messages)))

        for job_idx, messages in zip(job_idxs, list_of_messages):
            response = {
                "job_idx": job_idx,
                "messages": messages,
                "result": None,
                "text_response": None,
                "error": None,
                **self._submit_request(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    **kwargs,
                ),
            }

            yield Response(**response)

    def _submit_request(self, **kwargs) -> dict[str, Any]:
        """
        Submit a request to the VLLM API, populated by the given kwargs. Returns a dictionary of relevant
        keys (that will need to be merged with others to get the original messages and job idx etc).
        :param kwargs: kwargs to pass to the API
        :return: a dictionary with relevant response keys such as 'result' and 'text_response', or potentially
        'error' if something went wrong
        """
        payload = {
            "model": self.model_name,
            **kwargs,
        }
        response = {}
        try:
            completion = requests.post(
                self.endpoint, json=payload, headers={"Content-Type": "application/json"}
            ).json()
            choice = completion.choices[0]
            response["result"] = completion
            response["text_response"] = choice.message.content
        except Exception as exc:
            response["error"] = exc

        return response
