import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

from dutch_data.utils import Response


@dataclass
class TextGenerator(ABC):
    @abstractmethod
    def query_messages(self, messages, **kwargs) -> Response:
        raise NotImplementedError

    @abstractmethod
    def batch_query_messages(self, list_of_messages, **kwargs) -> Generator[Response, None, None]:
        raise NotImplementedError

    @staticmethod
    def _verify_list_of_messages(list_of_messages):
        if (
            not isinstance(list_of_messages, list)
            or not isinstance(list_of_messages[0], (tuple, list))
            or not isinstance(list_of_messages[0][0], (int, dict))
        ):
            example_input = [
                [{"role": "system", "content": "You are a good AI"}, {"role": "user", "content": "hi"}],
                {"role": "user", "content": "good morning"},
            ]
            pretty_example = json.dumps(example_input, indent=2)
            raise ValueError(
                f"list_of_messages is expected to be a list of conversations, where each conversation is a list of"
                f" messages. For instance,\n{pretty_example}"
                f"\nFor advanced use cases, you could also add job idxs yourself by passing a list of tuples of"
                f" (job_idx, messages) where messages is still expected to be a list of dicts (or compatible)."
            )

    @staticmethod
    def _verify_extra_args(args, list_of_messages):
        batch_size = len(list_of_messages)
        for arg_idx, arg in enumerate(args, 1):
            if len(arg) != batch_size:
                raise ValueError(
                    f"Every given extra arg must be a list of the same length as the number of items in the batch. "
                    f" So you could for instance pass a list of job hash identifiers as an extra arg (one for each"
                    f" item in the batch), followed by a list of speaker IDs (one for each item in the batch)."
                    f"Got {len(arg)} elements in extra argument no. {arg_idx} instead of {batch_size}."
                )
