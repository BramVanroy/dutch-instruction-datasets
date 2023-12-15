from pathlib import Path

import pytest
from dutch_data.azure_utils.credentials import Credentials


class TestCredentials:
    def setup_method(self):
        # Assumes that the credentials file is in the root directory of the project
        self.credentials_file = Path(__file__).parent / "dummy-credentials.json"

    def test_from_file_no_profile(self):
        # Will use the first profile
        creds = Credentials.from_json(self.credentials_file)
        assert isinstance(creds, Credentials)

    def test_from_file(self):
        # Assumes that the credentials file has a profile called "gpt-42-dummy"
        Credentials.from_json(self.credentials_file, "gpt-42-dummy")

    def test_invalid_profile(self):
        with pytest.raises(KeyError):
            Credentials.from_json(self.credentials_file, "thIsKeyDoesNotExist")

    def test_invalid_credentials_file(self):
        with pytest.raises(FileNotFoundError):
            Credentials.from_json("ThisFileDoesNotExist.json")
