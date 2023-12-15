from pathlib import Path

import openai
import pytest
import transformers
from dutch_data.azure_utils import AzureQuerier, Credentials
from dutch_data.text_generator import AzureTextGenerator
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage


transformers.logging.set_verbosity_error()


def _create_completion(finish_reason: str):
    return ChatCompletion(
        id="random-id",
        choices=[
            Choice(
                finish_reason=finish_reason,
                index=0,
                message=ChatCompletionMessage(
                    content=None
                    if finish_reason == "content_filter"
                    else "Life is complicated. I do not know what it means.",
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
    def mock_create(*args, **kwargs):
        input_text = kwargs["messages"][0]["content"]

        # Mocked response based on input_text
        if input_text == "stop":
            response = _create_completion("stop")
        elif input_text == "content_filter":
            response = _create_completion("content_filter")
        elif input_text == "exception":
            # Mocked exception - always raise an exception
            raise Exception("Mocked exception")

        return response

    # Patch the API call with the mock
    monkeypatch.setattr(openai.resources.chat.completions.Completions, "create", mock_create)


@pytest.fixture()
def azure_querier():
    # Assumes that the credentials file is in the root directory of the project
    credentials_file = Path(__file__).parent / "dummy-credentials.json"
    credentials = Credentials.from_json(credentials_file)
    return AzureQuerier.from_credentials(credentials, timeout=1, max_retries=1)


@pytest.fixture()
def azure_generator(azure_querier):
    return AzureTextGenerator(azure_querier)
