from dataclasses import dataclass
from typing import Generator

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

    def query_messages(
        self,
        messages: list[dict[str, str]] | tuple[int, list[dict[str, str]]],
        *args,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_new_tokens: int = 128,
        batch_size: int = 16,
        **other_gen_kwargs,
    ) -> Response:
        """
        Query the VLLM model with a single sample of messages.

        :param messages: messages to query the model with. A list of dicts where each dict must have a "role" and
        "content" keys
        :param args: any extra arguments to send, e.g. job hash identifiers, which will be include in the Response's
        extra_args attribute
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
        """
        args = [[arg] for arg in args]
        return next(
            self.batch_query_messages(
                [messages],
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                *args,
                **other_gen_kwargs,
            )
        )

    def batch_query_messages(
        self,
        list_of_messages: list[list[dict[str, str]]] | list[tuple[int, list[dict[str, str]]], ...],
        *args,
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

        :param list_of_messages: list of lists of messages to send to the API. We will add job idxs automatically
        but for more control you can also pass a list of tuples of (job_idx, messages) to specify the job idxs yourself
        :param args: any extra arguments to send, e.g. job hash identifiers, which will be include in the Response's
        extra_args attribute. Must be a list of lists where the sublists are of the same length as the number of items
        in the batch.
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
        self._verify_list_of_messages(list_of_messages)
        self._verify_extra_args(args, list_of_messages)

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
                    "extra_args": tuple([arg[item_idx] for arg in args]) if args else None,
                }

                yield Response(**response)

                item_idx += 1
