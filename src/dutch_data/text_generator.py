from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Generator, Literal

import torch
from dutch_data.azure_utils import AzureQuerier, Credentials
from dutch_data.azure_utils.querier import CyclicalAzureQuerier
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
    model_name: str
    device_map: dict[str, str | int | torch.device] | str | int | torch.device = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: torch.dtype | Literal["auto"] | None = None
    chat_template: str | None = None
    pipe: Pipeline = field(default=None, init=False)

    def __post_init__(self):
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
        messages: list[dict[str, str]],
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
        :return: generated assistant response
        """
        response = {
            "job_idx": 0,
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
        list_of_messages: list[list[dict[str, str]]],
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
        :return: generated assistant response. If it's a single message, return a string, otherwise a list of strings
        """
        # Interestingly, the pipeline will only yield a generator if the input was a generator or a Dataset
        # and they HAVE to be a Conversation
        generator_of_msgs = (Conversation(msgs) for msgs in list_of_messages)
        for item_idx, conversation in enumerate(
            self.pipe(
                generator_of_msgs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                batch_size=batch_size,
                **kwargs,
            )
        ):
            messages = list_of_messages[item_idx]
            yield Response(
                job_idx=item_idx,
                messages=messages,
                result=conversation,
                text_response=conversation.messages[-1]["content"],
            )


@dataclass
class AzureTextGenerator(TextGenerator):
    querier: AzureQuerier | CyclicalAzureQuerier

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
        :return: generated assistant response. If it's a single message, return a string, otherwise a list of strings
        """
        response = self.querier.query_messages(messages, **kwargs)

        return response

    def batch_query_messages(
        self,
        list_of_messages: list[list[dict[str, str]]],
        return_in_order: bool = True,
        **kwargs,
    ) -> Generator[Response, None, None]:
        """
        Query the model with the given batch of messages.
        :param list_of_messages: messages to query the model with. A list of lists of dicts where each dict must have
         a "role" and "content" keys
        :param return_in_order: whether to return the results in the order of the input
        :param kwargs: any keyword arguments to pass to the API
        :return: generated assistant response. If it's a single message, return a string, otherwise a list of strings
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
        querier = AzureQuerier(
            **asdict(credentials), max_retries=max_retries, timeout=timeout, max_workers=max_workers, verbose=verbose
        )
        return cls(querier)
