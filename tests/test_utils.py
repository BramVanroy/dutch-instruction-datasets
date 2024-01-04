import pytest
from dutch_data.utils import batchify, dict_to_tuple, extract_conversation_from_string


class TestUtils:
    def test_dict_to_tuple(self):
        test_dict = {"key2": "value2", "key3": "value3", "key1": "value1"}
        expected_result_sort = (("key1", "value1"), ("key2", "value2"), ("key3", "value3"))
        assert dict_to_tuple(test_dict) == expected_result_sort

        expected_result_notsorted = (("key2", "value2"), ("key3", "value3"), ("key1", "value1"))
        assert dict_to_tuple(test_dict, do_sort=False) == expected_result_notsorted

    def test_batchify(self):
        # Test with a list of 10 items and batch size of 3
        items = list(range(10))
        batch_size = 3
        expected_batches = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        assert batchify(items, batch_size) == expected_batches

        # Test with a list of 5 items and batch size of 2
        items = list(range(5))
        batch_size = 2
        expected_batches = [[0, 1], [2, 3], [4]]
        assert batchify(items, batch_size) == expected_batches

        # Test with a list of 5 items and batch size of 5 (i.e., the batch size is equal to the list size)
        items = list(range(5))
        batch_size = 5
        expected_batches = [items]
        assert batchify(items, batch_size) == expected_batches

    @pytest.mark.parametrize("drop_last_if_not_assistant", [True, False])
    def test_extract_conversation_from_string(self, drop_last_if_not_assistant):
        # Test with a conversation string where the last message is from the user
        conversation_string = "gebruiker: Hello\nassistent: Hi\ngebruiker: How are you?"
        expected_messages_drop = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        expected_messages_no_drop = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]

        messages = extract_conversation_from_string(
            conversation_string,
            user_id="gebruiker:",
            assistant_id="assistent:",
            drop_last_if_not_assistant=drop_last_if_not_assistant,
        )
        if drop_last_if_not_assistant:
            assert messages == expected_messages_drop
        else:
            assert messages == expected_messages_no_drop
