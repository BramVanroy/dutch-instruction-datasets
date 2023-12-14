import types
import unittest
from transformers import ConversationalPipeline
from dutch_data.text_generator import HFTextGenerator
from dutch_data.utils import Response


class TestHFTextGenerator(unittest.TestCase):
    def setUp(self):
        self.model_name = "microsoft/DialoGPT-small"
        self.generator = HFTextGenerator(self.model_name, device_map="auto")

    def test_initialization(self):
        self.assertIsInstance(self.generator.pipe, ConversationalPipeline)

    def test_query_messages(self):
        messages = [{"role": "system", "content": "You're a good assistant!"}, {"role": "user", "content": "What is the meaning of 42?"}]
        response = self.generator.query_messages(messages)
        self.assertIsInstance(response, Response)

    def test_batch_query_messages(self):
        list_of_messages = [[{"role": "system", "content": "You're a good assistant!"}, {"role": "user", "content": "What is the meaning of 42?"}],
                                    [{"role": "user", "content": "What is the meaning of life?"}]]
        response_generator = self.generator.batch_query_messages(list_of_messages)
        self.assertIsInstance(response_generator, types.GeneratorType)

        # Test Response objects
        for response_idx, response in enumerate(response_generator):
            self.assertIsInstance(response, Response)
            self.assertIsInstance(response.job_idx, int)
            self.assertEqual(response.job_idx, response_idx)
            self.assertIsNotNone(response.result)
            self.assertIsNotNone(response.text_response)


if __name__ == "__main__":
    unittest.main()
