import json
from pathlib import Path
from typing import Annotated

import typer
from colorama import Back, Fore, Style
from dutch_data import AzureQuerier, Credentials
from typer import Argument


app = typer.Typer()


def build_message(role: str, content: str) -> dict[str, str]:
    """
    Build a single message dictionary for the API
    """
    return {
        "role": role,
        "content": content,
    }


@app.command()
def interactive_playground(
    credentials_file: Annotated[str, typer.Argument(help="JSON file containing credentials")],
    credentials_profile: Annotated[
        str, Argument(help="which credential profile (key) to use from the credentials file")
    ],
):
    """
    Interactive playground for querying the API with given credentials and profile
    """
    credentials = Credentials.from_json(credentials_file, credentials_profile)
    querier = AzureQuerier.from_credentials(credentials)

    messages = []
    print("Enter a message/reply. Enter 'exit' to quit. Enter 'restart' to clear message history.")
    while user_query := input(f"{Back.CYAN}{Fore.WHITE}User:{Style.RESET_ALL} "):
        if user_query == "exit":
            break
        elif user_query == "restart":
            messages = []
            print()
            continue

        user_message = build_message("user", user_query)
        messages.append(user_message)

        response = next(querier.query_list_of_messages([messages])).result
        print(f"{Back.MAGENTA}{Fore.WHITE}Assistant:{Style.RESET_ALL} ", response)
        print()

        messages.append(build_message("assistant", response))
