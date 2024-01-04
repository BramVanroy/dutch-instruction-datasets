from dataclasses import dataclass
from typing import Any, Generator

from dutch_data.text_generator.base import TextGenerator
from dutch_data.utils import Response, batchify


try:
    import requests
    from vllm import LLM as VllmLLM
    from vllm import SamplingParams as VllmSamplingParams

    VLLM_AVAILABLE = True
except ModuleNotFoundError:
    VLLM_AVAILABLE = False


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
        if not VLLM_AVAILABLE:
            raise ModuleNotFoundError(
                "VLLM is not available. Please install VLLM from https://github.com/vllm-project/vllm/."
                " Note that Windows might not be supported."
            )

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
        use_tqdm: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_new_tokens: int = 128,
        batch_size: int = 16,
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
        :param presence_penalty: Float that penalizes new tokens based on whether they appear in the generated text
        so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
        :param frequency_penalty: Float that penalizes new tokens based on their frequency in the generated text so far.
        Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
        :param repetition_penalty: Float that penalizes new tokens based on whether they appear in the prompt and the
        generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage
        the model to repeat tokens.
        :param temperature: Float that controls the randomness of the sampling. Lower values make the model more
        deterministic, while higher values make the model more random. Zero means greedy sampling.
        :param top_p: Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set
        to 1 to consider all tokens.
        :param top_k: Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.
        :param max_new_tokens: max. number of tokens to generate
        :param batch_size: pseudo-batch size. While VLLM does batching automatically based on available RAM,
        we also add an option here for pseudo-batching. This is useful to get a better indication of process
        because VLLM only returns the final list but we would like to have more frequent return values instead of
        just at the end.
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
            max_tokens=max_new_tokens,
            **other_gen_kwargs,
        )

        item_idx = 0
        for batch_msgs in batchify(prompted_list_of_msgs, batch_size=batch_size):
            batch_generated = self.llm.generate(batch_msgs, sampling_params, use_tqdm=use_tqdm)
            for generated, item_messages in zip(batch_generated, batch_msgs):
                job_idx = job_idxs[item_idx]
                generated_text = generated.outputs[0].text
                response = {
                    "job_idx": job_idx,
                    "messages": item_messages,
                    "result": generated,
                    "text_response": generated_text,
                    "error": None,
                }

                yield Response(**response)

                item_idx += 1


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
