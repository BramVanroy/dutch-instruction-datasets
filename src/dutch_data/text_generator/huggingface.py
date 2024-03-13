import logging
from dataclasses import dataclass, field
from typing import Generator, Literal

import torch
from dutch_data.text_generator.base import TextGenerator
from dutch_data.utils import Response
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Conversation, Pipeline, pipeline


if (
    torch.cuda.is_available()
    and not torch.backends.cuda.matmul.allow_tf32
    and torch.cuda.get_device_capability() >= (8, 0)
):
    torch.set_float32_matmul_precision("high")


@dataclass
class HFTextGenerator(TextGenerator):
    """
    Text generator that uses locally loaded conversational pipelines.
    :param model_name: name of the model to use
    :param device_map: device map to use for the pipeline
    :param load_in_8bit: whether to load the model in 8bit precision to save memory
    :param load_in_4bit: whether to load the model in 4bit precision to save memory
    :param use_flash_attention: whether to use the flash attention implementation
    :param trust_remote_code: whether to trust remote code or not. Required for some (newer) models
    :param torch_dtype: data type to use for the model, e.g. 'float16' or 'auto'
    :param chat_template: Jinja template to use for the chat if you want to overwrite the default template that comes
    with a given model. See https://huggingface.co/transformers/main_classes/pipelines.html#transformers.ConversationalPipeline
    """

    model_name: str
    device_map: dict[str, str | int | torch.device] | str | int | torch.device = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = False
    trust_remote_code: bool = False
    torch_dtype: torch.dtype | Literal["auto"] | str | None = None
    chat_template: str | None = None
    pipe: Pipeline = field(default=None, init=False)

    def __post_init__(self):
        if self.torch_dtype is not None and self.torch_dtype != "auto":
            try:
                self.torch_dtype = getattr(torch, self.torch_dtype)
            except AttributeError:
                raise ValueError(f"Invalid torch dtype: {self.torch_dtype}")

        if self.use_flash_attention:
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                logging.error("Successfully enabled torch's enable_flash_sdp")
            except Exception as exc:
                logging.error(
                    "Could not enable flash attention with torch.backends.cuda.enable_flash_sdp."
                    f" Disabling...\nError message: {exc}"
                )
                pass

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self.torch_dtype,
        )
        model_kwargs = {
            "quantization_config": bnb_config,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": True,
            "device_map": self.device_map,
        }

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        try:
            model = torch.compile(model)
            logging.error("Successfully compiled model with torch.compile")
        except RuntimeError:
            pass

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.pipe = pipeline(
            "conversational",
            model=model,
            tokenizer=tokenizer,
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
        **other_gen_kwargs,
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
        :param other_gen_kwargs: additional kwargs to pass to the pipeline call
        :return: generated assistant Response
        """
        args = [[arg] for arg in args]
        return next(
            self.batch_query_messages(
                [messages],
                *args,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **other_gen_kwargs,
            )
        )

    @torch.inference_mode()
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
        **other_gen_kwargs,
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
        :param other_gen_kwargs: additional kwargs to pass to the pipeline call
        :return: generator of Responses
        """
        self._verify_list_of_messages(list_of_messages)
        self._verify_extra_args(args, list_of_messages)

        if isinstance(list_of_messages[0][0], int):
            job_idxs = [item[0] for item in list_of_messages]
            list_of_messages: list[list[dict[str, str]]] = [item[1] for item in list_of_messages]  # noqa
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
                    **other_gen_kwargs,
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
