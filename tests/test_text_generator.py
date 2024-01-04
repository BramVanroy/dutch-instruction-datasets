import types

from dutch_data.utils import Response


class TestTextGenerators:
    def test_query_messages(self, text_generator):
        messages = [{"role": "user", "content": "stop"}]
        response = text_generator.query_messages(messages)

        assert isinstance(response, Response)
        assert isinstance(response.job_idx, int)
        assert response.job_idx == 0
        assert response.result is not None
        assert response.text_response is not None

    def test_batch_query_messages(self, text_generator):
        list_of_messages = [
            [{"role": "user", "content": "stop"}],
            [{"role": "user", "content": "stop"}],
        ]
        response_generator = text_generator.batch_query_messages(list_of_messages)
        assert isinstance(response_generator, types.GeneratorType)

        for response_idx, response in enumerate(response_generator):
            assert isinstance(response, Response)
            assert isinstance(response.job_idx, int)
            assert response.job_idx == response_idx
            assert response.result is not None
            assert response.text_response is not None

    def test_extra_args_query_messages(self, text_generator):
        messages = [{"role": "user", "content": "stop"}]

        # Single extra_arg: even with one extra_arg, the output will be an iterable (tuple here)
        extra_arg = "extra_arg_value1"
        response = text_generator.query_messages(messages, extra_arg)
        assert response.extra_args == tuple([extra_arg])

        # Multiple extra_args
        extra_args = ["extra_arg_value1", "extra_arg_value2", "extra_arg_value3"]
        response = text_generator.query_messages(messages, *extra_args)
        assert response.extra_args == tuple(extra_args)

    def test_extra_args_query_list_of_messages(self, text_generator):
        list_of_messages = [
            [{"role": "user", "content": "stop"}],
            [{"role": "user", "content": "stop"}],
        ]

        # Passing one list of extra_args, meaning one for the first message and one for the second
        extra_args = [["msg_1_extra_arg", "msg_2_extra_arg"]]
        response_generator = text_generator.batch_query_messages(list_of_messages, *extra_args)
        for i, response in enumerate(response_generator):
            assert response.extra_args == tuple([extra[i] for extra in extra_args])

        # Passing multiple lists of extra_args, so the first sublist will be the extra args of the first item
        # while the second sublist will be the extra args of the second item
        extra_args = [
            ["msg1_extra_args_value1", "msg2_extra_args_value1"],
            ["msg1_extra_args_value2", "msg2_extra_args_value2"],
            ["msg1_extra_args_value3", "msg2_extra_args_value3"],
        ]
        response_generator = text_generator.batch_query_messages(list_of_messages, *extra_args)
        for i, response in enumerate(response_generator):
            assert response.extra_args == tuple([extra[i] for extra in extra_args])
