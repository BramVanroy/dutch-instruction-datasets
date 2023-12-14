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
    def from_json(cls, credentials_file: str | PathLike, credentials_profile: str | None = None):
        """
        Load credentials from a JSON file. If no profile is given, the first one in the file will be used.
        """
        credentials = json.loads(Path(credentials_file).read_text(encoding="utf-8"))
        try:
            credentials = credentials[credentials_profile]
        except KeyError:
            credentials = list(credentials.values())[0]

        return cls(**credentials)
