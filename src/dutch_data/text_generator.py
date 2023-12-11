from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import torch
from dutch_data import AzureQuerier
from openai.types.chat import ChatCompletion
from transformers import Pipeline, pipeline


@dataclass
class TextGenerator(ABC):
    @abstractmethod
    def query_messages(self, messages, **kwargs):
        raise NotImplementedError


@dataclass
class HFTextGenerator(TextGenerator):
    model_name: str
    device_map: dict[str, str | int | torch.device] | str | int | torch.device = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: torch.dtype | Literal["auto"] | None = None
    pipe: Pipeline = field(default=None, init=False)

    def __post_init__(self):
        model_kwargs = {
            "device_map": self.device_map,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "torch_dtype": self.torch_dtype,
        }
        self.pipe = pipeline(
            "text-generation",
            model=self.modelname,
            model_kwargs=model_kwargs,
        )

    def query_messages(
        self,
        messages: list[dict[str, str]] | list[list[dict[str, str]]],
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        chat_template: str | None = None,
        return_full_text: bool = False,
        batch_size: int = 1,
        **kwargs,
    ) -> str | list[str]:
        """
        Query the model with the given messages.
        :param messages: messages to query the model with. A list of dicts where each dict must have a "role" and
        "content" keys in the case of single querying, or a list of such lists in the case of batch querying.
        :param max_new_tokens: max new tokens to generate
        :param do_sample: whether to use sampling or not
        :param temperature: sampling temperature
        :param top_k: k for top-k sampling
        :param top_p: p for top-p sampling
        :param chat_template: Jinja template for the chat structure. Will use the model's default but you can
        override it here or add another one if the model doesn't have one
        :param return_full_text: whether to return the full text (incl. prompt) or just the generated text
        :param batch_size: batch size to use for the pipeline call
        :param kwargs: additional kwargs to pass to the pipeline call
        :return: generated assistant response. If it's a single message, return a string, otherwise a list of strings
        """
        if "num_return_sequences" in kwargs:
            raise ValueError(
                "num_return_sequences is not supported when using the TextGenerator class."
            )

        if isinstance(messages[0], dict):
            messages = [messages]

        prompted_batch_messages = [
            self.pipe.tokenizer.apply_chat_template(
                msg, chat_template=chat_template, tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]

        generated = self.pipe(
            prompted_batch_messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_full_text=return_full_text,
            batch_size=batch_size,
            **kwargs,
        )

        # The outputs have an extra dimension because in the pipeline it is possible to return multiple sequences
        if len(generated) == 1:
            return generated[0][0]["generated_text"]

        return [gen[0]["generated_text"] for gen in generated]


@dataclass
class AzureTextGenerator(TextGenerator):
    querier: AzureQuerier

    def query_messages(
        self,
        messages: list[dict[str, str]] | list[list[dict[str, str]]],
        return_full_api_output: bool = False,
        return_in_order: bool = True,
        **kwargs,
    ) -> str | list[str] | ChatCompletion | list[ChatCompletion]:
        """
        Query the model with the given messages.
        :param messages: messages to query the model with. A list of dicts where each dict must have a "role" and
        "content" keys in the case of single querying, or a list of such lists in the case of batch querying.
        :param return_full_api_output: whether to return the full API output or only the generated text
        :param return_in_order: whether to return the results in the order of the input
        :param kwargs: any keyword arguments to pass to the API
        :return: generated assistant response. If it's a single message, return a string, otherwise a list of strings
        """
        if isinstance(messages[0], dict):
            messages = [messages]

        responses = list(self.querier.query_list_of_messages(messages, **kwargs))

        if len(responses) == 1:
            return responses[0].result

        return [response.result for response in responses]
