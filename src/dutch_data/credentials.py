from dataclasses import dataclass


@dataclass
class Credentials:
    """
    Credentials for Azure OpenAI API.
    """

    api_key: str
    api_version: str
    deployment_name: str
    endpoint: str
