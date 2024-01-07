from pathlib import Path

import pytest
from dutch_data.azure_utils.credentials import Credentials
from dutch_data.text_generator import AzureTextGenerator


class TestAzureCredentials:
    def setup_method(self):
        # Assumes that the credentials file is in the root directory of the project
        self.credentials_file = Path(__file__).parent / "dummy-credentials.json"

    def test_from_file_no_profile(self):
        # Will use the first profile
        creds = Credentials.from_json(self.credentials_file)
        assert isinstance(creds, Credentials)
        assert creds.azure_deployment == "DeployedDummy1"

    def test_from_file(self):
        # Assumes that the credentials file has a profile called "gpt-42-dummy"
        Credentials.from_json(self.credentials_file, "gpt-42-dummy")

    def test_invalid_profile(self):
        with pytest.raises(KeyError):
            Credentials.from_json(self.credentials_file, "thIsKeyDoesNotExist")

    def test_invalid_credentials_file(self):
        with pytest.raises(FileNotFoundError):
            Credentials.from_json("ThisFileDoesNotExist.json")


class TestAzureGenerator:
    def setup_method(self):
        # Assumes that the credentials file is in the root directory of the project
        self.credentials_file = Path(__file__).parent / "dummy-credentials.json"
        self.credentials = Credentials.from_json(self.credentials_file)

    def test_multi_from_json(self):
        azure_generator = AzureTextGenerator.from_json(self.credentials_file)

        assert isinstance(azure_generator, AzureTextGenerator)
        assert len(azure_generator.clients) == 2

    def test_single_from_json(self):
        azure_generator = AzureTextGenerator.from_json(self.credentials_file, "gpt-42-dummy")

        assert len(azure_generator.clients) == 1

    def test_single_from_credentials(self):
        azure_generator = AzureTextGenerator.from_credentials(self.credentials)

        assert len(azure_generator.clients) == 1

    def test_from_json_invalid_args(self):
        with pytest.raises(FileNotFoundError):
            AzureTextGenerator.from_json("ThisFileDoesNotExist.json")

        with pytest.raises(KeyError):
            AzureTextGenerator.from_json(self.credentials_file, "ThisKeyDoesNotExist")

        with pytest.raises(ValueError):
            AzureTextGenerator.from_json(self.credentials_file, timeout=-1)

        with pytest.raises(ValueError):
            AzureTextGenerator.from_json(self.credentials_file, max_retries=-1)

    def test_from_credentials_invalid_args(self):
        with pytest.raises(TypeError):
            # credentials must be of type Credentials, not the credentials JSON file
            AzureTextGenerator.from_credentials(self.credentials_file, timeout=-1)

        with pytest.raises(ValueError):
            AzureTextGenerator.from_credentials(self.credentials, timeout=-1)

        with pytest.raises(ValueError):
            AzureTextGenerator.from_credentials(self.credentials, max_retries=-1)
