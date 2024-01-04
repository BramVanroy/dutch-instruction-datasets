from dataclasses import dataclass, field
from typing import Generator, Literal

import torch
from dutch_data.text_generator.base import TextGenerator
from dutch_data.utils import Response
from transformers import Conversation, Pipeline, pipeline


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
        *args,
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
        :param args: any extra arguments to send, e.g. job hash identifiers, which will be include in the Response's
        extra_args attribute
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
            "extra_args": args if args else None,
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
        *args,
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
        :param list_of_messages: list of lists of messages to process. We will add job idxs automatically
        but for more control you can also pass a list of tuples of (job_idx, messages) to specify the job idxs yourself
        :param args: any extra arguments to send, e.g. job hash identifiers, which will be include in the Response's
        extra_args attribute. Must be a list of lists where the sublists are of the same length as the number of items
        in the batch.
        :param max_new_tokens: max new tokens to generate
        :param do_sample: whether to use sampling or not
        :param temperature: sampling temperature
        :param top_k: k for top-k sampling
        :param top_p: p for top-p sampling
        :param batch_size: batch size for the pipeline
        :param kwargs: additional kwargs to pass to the pipeline call
        :return: generator of Responses
        """
        self._verify_list_of_messages(list_of_messages)
        self._verify_extra_args(args, list_of_messages)

        if isinstance(list_of_messages[0][0], int):
            job_idxs = [item[0] for item in list_of_messages]
            list_of_messages = [item[1] for item in list_of_messages]
        else:
            job_idxs = list(range(len(list_of_messages)))

        # Interestingly, the pipeline will only yield a generator if the input was a generator or a Dataset
        # and they HAVE to be a Conversation
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
                extra_args=tuple([arg[item_idx] for arg in args]) if args else None,
            )
