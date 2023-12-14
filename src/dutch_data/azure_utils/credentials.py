import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path


@dataclass
class Credentials:
    """
    Credentials for Azure OpenAI API.
    """

    api_key: str
    api_version: str
    deployment_name: str
    endpoint: str

    @classmethod
    def from_json(cls, credentials_file: str | PathLike, credentials_profile: str):
        """
        Load credentials from a JSON file.
        """
        credentials = json.loads(Path(credentials_file).read_text(encoding="utf-8"))
        try:
            credentials = credentials[credentials_profile]
        except KeyError:
            raise KeyError(f"Credentials profile {credentials_profile} not found in credentials file")

        return cls(**credentials)
