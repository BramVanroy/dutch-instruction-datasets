import types

from dutch_data.utils import Response


class TestAzureTextGenerator:
    # Thorough testing of the internal querier is done in the querier tests
    def test_query_messages(self, azure_generator):
        messages = [{"role": "user", "content": "stop"}]
        response = azure_generator.query_messages(messages)
        assert isinstance(response, Response)

    def test_batch_query_messages(self, azure_generator):
        list_of_messages = [
            [{"role": "user", "content": "stop"}],
            [{"role": "user", "content": "stop"}],
        ]
        response_generator = azure_generator.batch_query_messages(list_of_messages)
        assert isinstance(response_generator, types.GeneratorType)
        assert isinstance(next(response_generator), Response)
