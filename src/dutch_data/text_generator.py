from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from dutch_data import AzureQuerier
from transformers import Pipeline, pipeline


@dataclass
class TextGenerator(ABC):
    def prompt_message(self):
        pass

    @abstractmethod
    def query_messages(self, messages, **kwargs):
        raise NotImplementedError


@dataclass
class HFTextGenerator(TextGenerator):
    model_name: str
    device_map: dict[str, str | int | torch.device] | str | int | torch.device = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str | None = None
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
            model=self.model_name,
            model_kwargs=model_kwargs,
        )

    def query_messages(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        chat_template: str | None = None,
        return_full_text: bool = False,
        **kwargs,
    ) -> str:
        """
        Query the model with the given messages.
        :param messages: messages to query the model with. A list of dicts where each dict must have a "role" and
        "content" keys
        :param max_new_tokens: max new tokens to generate
        :param do_sample: whether to use sampling or not
        :param temperature: sampling temperature
        :param top_k: k for top-k sampling
        :param top_p: p for top-p sampling
        :param chat_template: Jinja template for the chat structure. Will use the model's default but you can
        override it here or add another one if the model doesn't have one
        :param return_full_text: whether to return the full text (incl. prompt) or just the generated text
        :param kwargs: additional kwargs to pass to the pipeline call
        :return: generated assistant response
        """
        prompted_messages = self.pipe.tokenizer.apply_chat_template(
            messages, chat_template=chat_template, tokenize=False, add_generation_prompt=True
        )
        generated = self.pipe(
            prompted_messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_full_text=return_full_text,
            **kwargs,
        )
        return generated[0]["generated_text"]
