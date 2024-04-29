import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path


@dataclass
class AzureCredentials:
    """
    Credentials for Azure OpenAI API.
    """

    api_key: str
    api_version: str
    azure_deployment: str
    azure_endpoint: str

    @classmethod
    def from_json(cls, credentials_file: str | PathLike, credentials_profile: str | None = None):
        """
        Load credentials from a JSON file. If no profile is given, the first one in the file will be used.
        """
        pfcredentials_file = Path(credentials_file).resolve()

        if not pfcredentials_file.exists():
            raise FileNotFoundError(f"Credentials file {pfcredentials_file} does not exist.")

        credentials = json.loads(pfcredentials_file.read_text(encoding="utf-8"))
        credentials = credentials[credentials_profile] if credentials_profile else list(credentials.values())[0]

        return cls(**credentials)


@dataclass
class OpenAiCredentials:
    """
    Credentials for OpenAI API.
    """

    api_key: str
    model: str

    @classmethod
    def from_json(cls, credentials_file: str | PathLike, credentials_profile: str | None = None):
        """
        Load credentials from a JSON file. If no profile is given, the first one in the file will be used.
        """
        pfcredentials_file = Path(credentials_file).resolve()

        if not pfcredentials_file.exists():
            raise FileNotFoundError(f"Credentials file {pfcredentials_file} does not exist.")

        credentials = json.loads(pfcredentials_file.read_text(encoding="utf-8"))
        credentials = credentials[credentials_profile] if credentials_profile else list(credentials.values())[0]

        return cls(**credentials)
