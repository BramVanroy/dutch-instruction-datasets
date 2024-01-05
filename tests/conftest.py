from pathlib import Path
from typing import Literal


import openai
import pytest
import torch
import transformers
import datasets
from dutch_data.azure_utils.credentials import Credentials
from dutch_data.text_generator import AzureTextGenerator, HFTextGenerator, VLLMTextGenerator
from dutch_data.text_generator.vllm import VLLM_AVAILABLE
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pytest_lazyfixture import lazy_fixture


transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()




def _create_completion(finish_reason: Literal["content_filter", "exception", "stop"] = "stop", text: str | None = None):
    if finish_reason == "stop":
        text = text if text else "Life is complicated. I do not know what it means."
    else:
        text = None

    return ChatCompletion(
        id="random-id",
        choices=[
            Choice(
                finish_reason=finish_reason,
                logprobs=None,
                index=0,
                message=ChatCompletionMessage(
                    content=text,
                    role="assistant",
                ),
            )
        ],
        created=1234567890,
        model="some-model",
        object="chat.completion",
    )


@pytest.fixture(autouse=True)
def mock_openai_completions_create(monkeypatch):
    """
    We're mocking the OpenAI API call to completions.create() to avoid incurring costs.
    We're mimicking three scenarios based on the first "content" message in the request:
    - "stop": return a completion with finish_reason="stop"
    - "content_filter": return a completion with finish_reason="content_filter"
    - "exception": raise an exception
    """

    def mock_create(*args, **kwargs):
        input_text = kwargs["messages"][0]["content"]

        # Mocked response based on input_text
        if input_text == "content_filter":
            return _create_completion("content_filter")
        elif input_text == "exception":
            # Mocked exception - always raise an exception
            raise Exception("Mocked exception")
        else:  # In case of "stop" or any other input
            return _create_completion("stop", text=input_text)

    # Patch the API call with the mock
    monkeypatch.setattr(openai.resources.chat.completions.Completions, "create", mock_create)


TEXT_GENERATORS = {}


@pytest.fixture(params=["single_from_credentials", "multi"])
def azure_generator(request):
    credentials_file = Path(__file__).parent / "dummy-credentials.json"

    if request.param == "single_from_credentials":
        if "single_azure" not in TEXT_GENERATORS:
            # If no profile is used in Credentials, only the first one in the file will be used.
            credentials = Credentials.from_json(credentials_file)
            TEXT_GENERATORS["single_azure"] = AzureTextGenerator.from_credentials(
                credentials, timeout=1, max_retries=1
            )
        yield TEXT_GENERATORS["single_azure"]
    elif request.param == "multi":
        # If no profile is specified in the AzureQuerier, _all_ profiles in the file will be used,
        # switching between profiles at every new request.
        if "multi_azure" not in TEXT_GENERATORS:
            TEXT_GENERATORS["multi_azure"] = AzureTextGenerator.from_json(credentials_file, timeout=1, max_retries=1)
        yield TEXT_GENERATORS["multi_azure"]


@pytest.fixture
def hf_generator():
    if "huggingface" not in TEXT_GENERATORS:
        TEXT_GENERATORS["huggingface"] = HFTextGenerator("microsoft/DialoGPT-small", device_map="auto")

    return TEXT_GENERATORS["huggingface"]


@pytest.fixture
def vllm_generator():
    if VLLM_AVAILABLE:
        if "vllm" not in TEXT_GENERATORS:
            TEXT_GENERATORS["vllm"] = VLLMTextGenerator(
                "microsoft/DialoGPT-small", tensor_parallel_size=torch.cuda.device_count()
            )
        return TEXT_GENERATORS["vllm"]
    else:
        pytest.skip("VLLM not installed. Skipping it in tests.")


@pytest.fixture(params=[lazy_fixture("vllm_generator"), lazy_fixture("hf_generator"), lazy_fixture("azure_generator")])
def text_generator(request):
    yield request.param
