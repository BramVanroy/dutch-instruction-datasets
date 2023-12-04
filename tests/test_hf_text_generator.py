import unittest
from transformers import TextGenerationPipeline
from dutch_data.text_generator import HFTextGenerator


class TestHFTextGenerator(unittest.TestCase):
    def setUp(self):
        self.model_name = "HuggingFaceH4/zephyr-7b-beta"
        self.generator = HFTextGenerator(self.model_name)

    def test_initialization(self):
        self.assertIsInstance(self.generator.pipe, TextGenerationPipeline)

    def test_query_messages(self):
        messages = [{"role": "system", "content": "You're a good assistant!"}, {"role": "user", "content": "What is the meaning of 42?"}]
        output = self.generator.query_messages(messages)
        self.assertIsInstance(output, str)


if __name__ == "__main__":
    unittest.main()
