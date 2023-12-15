import types

import pytest
from dutch_data.azure_utils.querier import ContentFilterException
from dutch_data.utils import Response
from openai.types.chat.chat_completion import ChatCompletion


class TestAzureQuerier:
    @pytest.mark.parametrize("request_type", ["stop", "content_filter", "exception"])
    def test_query_messages(self, azure_querier, request_type):
        messages = (0, ((("role", "user"), ("content", request_type)),))
        response = azure_querier.query_messages(messages)
        assert isinstance(response, Response)

        assert response.job_idx == 0

        if request_type == "stop":
            assert isinstance(response.text_response, str)
            assert response.error is None
            assert isinstance(response.result, ChatCompletion)
        elif request_type == "content_filter":
            assert response.text_response is None
            assert isinstance(response.error, ContentFilterException)
            assert isinstance(response.result, ChatCompletion)
        elif request_type == "exception":
            assert response.text_response is None
            assert isinstance(response.error, Exception)
            assert response.result is None

    @pytest.mark.parametrize("request_type", ["stop", "content_filter", "exception"])
    def test_query_list_of_messages(self, azure_querier, request_type):
        list_of_messages = [[{"role": "user", "content": "stop"}], [{"role": "user", "content": request_type}]]
        response_generator = azure_querier.query_list_of_messages(list_of_messages)
        assert isinstance(response_generator, types.GeneratorType)

        # Test Response objects
        for response_idx, response in enumerate(response_generator):
            assert isinstance(response, Response)
            assert isinstance(response.job_idx, int)
            assert response.job_idx == response_idx

            if response_idx == 0:
                assert isinstance(response.text_response, str)
                assert response.error is None
                assert isinstance(response.result, ChatCompletion)
            elif response_idx == 1:
                if request_type == "stop":
                    assert isinstance(response.text_response, str)
                    assert response.error is None
                    assert isinstance(response.result, ChatCompletion)
                elif request_type == "content_filter":
                    assert response.text_response is None
                    assert isinstance(response.error, ContentFilterException)
                    assert isinstance(response.result, ChatCompletion)
                elif request_type == "exception":
                    assert response.text_response is None
                    assert isinstance(response.error, Exception)
                    assert response.result is None
