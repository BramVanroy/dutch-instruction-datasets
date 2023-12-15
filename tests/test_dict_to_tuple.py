from dutch_data.utils import dict_to_tuple


class TestDictToTuple:
    def test_dict_to_tuple(self):
        test_dict = {"key1": "value1", "key2": "value2", "key3": "value3"}
        expected_result = (("key1", "value1"), ("key2", "value2"), ("key3", "value3"))
        assert dict_to_tuple(test_dict) == expected_result
