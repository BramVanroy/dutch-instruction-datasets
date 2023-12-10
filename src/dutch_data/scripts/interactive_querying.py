from typing import Annotated, Optional

import typer
from colorama import Back, Fore, Style
from dutch_data import AzureQuerier, Credentials
from dutch_data.text_generator import AzureTextGenerator, HFTextGenerator, TextGenerator
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
def huggingface(
    model_name: Annotated[str, typer.Argument(help="the model name to use from HuggingFace's model hub (e.g. 'BramVanroy/Llama-2-13b-chat-dutch')")],
    device_map: Annotated[Optional[str], typer.Option(help="device (map) to use. You probably want to set this to 'auto' or to a specific device")] = None,
    load_in_8bit: Annotated[bool, typer.Option(help="whether to load the model in 8 bit precision")] = False,
    load_in_4bit: Annotated[bool, typer.Option(help="whether to load the model in 4 bit precision")] = False,
):
    """
    Interactive playground for querying model's that are available on HuggingFace's model hub
    """
    generator = HFTextGenerator(
        model_name=model_name,
        device_map=device_map,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )
    interactive_playground(generator)


@app.command()
def azure(
    credentials_file: Annotated[str, typer.Argument(help="JSON file containing credentials")],
    credentials_profile: Annotated[
        str, Argument(help="which credential profile (key) to use from the credentials file")
    ],
):
    """
    Interactive playground for querying the Azure API with given credentials and profile
    """
    credentials = Credentials.from_json(credentials_file, credentials_profile)
    querier = AzureQuerier.from_credentials(credentials)
    generator = AzureTextGenerator(querier)
    interactive_playground(generator)


def interactive_playground(
    generator: TextGenerator,
):
    """
    Interactive playground for querying the Azure API or HuggingFace models
    """
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

        response = generator.query_messages(messages)
        print(f"{Back.MAGENTA}{Fore.WHITE}Assistant:{Style.RESET_ALL} ", response)
        print()

        messages.append(build_message("assistant", response))
