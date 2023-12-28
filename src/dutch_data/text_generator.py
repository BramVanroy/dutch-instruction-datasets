from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Generator, Literal

import requests
import torch
from dutch_data.azure_utils import AzureQuerier, Credentials
from dutch_data.utils import Response
from transformers import Conversation, Pipeline, pipeline
from vllm import LLM as VllmLLM
from vllm import SamplingParams as VllmSamplingParams


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
        messages: list[dict[str, str]] | tuple[int, list[dict[str, str]]],
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
        list_of_messages: list[list[dict[str, str]]] | list[tuple[int, list[dict[str, str]]], ...],
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
    Text generator for the VLLM Python library.

    :param model_name: name of the model to use
    :param trust_remote_code: whether to trust code in the remote model's repository. Might be required for very
    new models
    :param tensor_parallel_size: tensor parallel size to use for the model, i.e., how many GPUs to use
    :param dtype: data type to use for the model, e.g. `float32`, `float16`, `bfloat16`, or 'auto'
    """

    model_name: str
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1
    dtype: str = "auto"

    def __post_init__(self):
        self.llm = VllmLLM(
            model=self.model_name,
            trust_remote_code=self.trust_remote_code,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
        )

    def query_messages(self, messages: list[dict[str, str]] | tuple[int, list[dict[str, str]]], **kwargs) -> Response:
        """
        Query the VLLM model with a single sample of messages.
        :param messages: messages to query the model with. A list of dicts where each dict must have a "role" and
        "content" keys, or a tuple of (job_idx, messages) where job_idx is an int
        :param kwargs: additional kwargs to pass to the model call, which include generation parameters such as
        'temperature', 'top_p', 'top_k', etc.
        :return: generated assistant Response
        """
        return next(self.batch_query_messages([messages]), **kwargs)

    def batch_query_messages(
        self,
        list_of_messages: list[list[dict[str, str]]] | list[tuple[int, list[dict[str, str]]], ...],
        use_tqdm: bool = True,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        **other_gen_kwargs,
    ) -> Generator[Response, None, None]:
        """
        Query the VLLM model with the given messages. Batching will occur automatically in VLLM based on memory
        constriants. Parameter descriptions taken from the VLLM documentation.

        :param list_of_messages: messages to query the model with. A list of lists of dicts where each dict must have
        a "role" and "content" keys, or a list of tuples of (job_idx, messages) where job_idx is an int
        :param use_tqdm: whether to use tqdm for progress bar presence_penalty: Float that penalizes new tokens based
        on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while
        values < 0 encourage the model to repeat tokens.
        frequency_penalty: Float that penalizes new tokens based on their frequency in the generated text so far.
        Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
        repetition_penalty: Float that penalizes new tokens based on whether they appear in the prompt and the
        generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage
        the model to repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower values make the model more
        deterministic, while higher values make the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set
        to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.
        :param other_gen_kwargs: other generation kwargs to pass to the generate function
        :return: generator of Responses
        """
        if isinstance(list_of_messages[0][0], int):
            job_idxs = [item[0] for item in list_of_messages]
            list_of_messages = [item[1] for item in list_of_messages]
        else:
            job_idxs = list(range(len(list_of_messages)))

        prompted_list_of_msgs = [
            self.llm.get_tokenizer().apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in list_of_messages
        ]

        sampling_params = VllmSamplingParams(
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **other_gen_kwargs,
        )

        generated = self.llm.generate(prompted_list_of_msgs, sampling_params, use_tqdm=use_tqdm)
        for job_idx, output in zip(job_idxs, generated):
            generated_text = output.outputs[0].text
            response = {
                "job_idx": job_idx,
                "messages": messages,
                "result": output,
                "text_response": generated_text,
                "error": None,
            }

            yield Response(**response)


@dataclass
class VLLMServerTextGenerator(TextGenerator):
    """
    Text generator for the VLLM API, which will call on an endpoint URL that has an VLLM instance running.

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
            choice = completion["choices"][0]
            response["result"] = completion
            response["text_response"] = choice["message"]["content"]
        except Exception as exc:
            response["error"] = exc

        return response
