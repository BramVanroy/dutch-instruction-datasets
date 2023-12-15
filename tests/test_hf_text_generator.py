import types

from dutch_data.text_generator import HFTextGenerator
from dutch_data.utils import Response
from transformers import ConversationalPipeline


class TestHFTextGenerator:
    def setup_method(self):
        self.model_name = "microsoft/DialoGPT-small"
        self.generator = HFTextGenerator(self.model_name, device_map="auto")

    def test_initialization(self):
        assert isinstance(self.generator.pipe, ConversationalPipeline)

    def test_query_messages(self):
        messages = [
            {"role": "system", "content": "You're a good assistant!"},
            {"role": "user", "content": "What is the meaning of 42?"},
        ]
        response = self.generator.query_messages(messages)
        assert isinstance(response, Response)

    def test_batch_query_messages(self):
        list_of_messages = [
            [
                {"role": "system", "content": "You're a good assistant!"},
                {"role": "user", "content": "What is the meaning of 42?"},
            ],
            [{"role": "user", "content": "What is the meaning of life?"}],
        ]
        response_generator = self.generator.batch_query_messages(list_of_messages)
        assert isinstance(response_generator, types.GeneratorType)

        # Test Response objects
        for response_idx, response in enumerate(response_generator):
            assert isinstance(response, Response)
            assert isinstance(response.job_idx, int)
            assert response.job_idx == response_idx
            assert response.result is not None
            assert response.text_response is not None
