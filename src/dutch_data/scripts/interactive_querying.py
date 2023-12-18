from typing import Annotated, Optional

import typer
from colorama import Back, Fore, Style
from dutch_data.azure_utils import AzureQuerier, Credentials
from dutch_data.text_generator import AzureTextGenerator, HFTextGenerator, TextGenerator
from dutch_data.utils import build_message
from typer import Argument


app = typer.Typer()


@app.command()
def huggingface(
    model_name: Annotated[
        str,
        typer.Argument(
            help="the model name to use from HuggingFace's model hub (e.g. 'BramVanroy/Llama-2-13b-chat-dutch')"
        ),
    ],
    system_prompt: Annotated[
        Optional[str], typer.Option("-p", "--system-prompt", help="an optional system message")
    ] = None,
    device_map: Annotated[
        Optional[str],
        typer.Option(help="device (map) to use. You probably want to set this to 'auto' or to a specific device"),
    ] = None,
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
    interactive_playground(generator, system_prompt=system_prompt)


@app.command()
def azure(
    credentials_file: Annotated[str, typer.Argument(help="JSON file containing credentials")],
    credentials_profile: Annotated[
        str, Argument(help="which credential profile (key) to use from the credentials file")
    ],
    system_prompt: Annotated[
        Optional[str], typer.Option("-p", "--system-prompt", help="an optional system message")
    ] = None,
):
    """
    Interactive playground for querying the Azure API with given credentials and profile
    """
    credentials = Credentials.from_json(credentials_file, credentials_profile)
    querier = AzureQuerier.from_credentials(credentials)
    generator = AzureTextGenerator(querier)
    interactive_playground(generator, system_prompt=system_prompt)


def interactive_playground(
    generator: TextGenerator,
    system_prompt: str | None = None,
):
    """
    Interactive playground for querying the Azure API or HuggingFace models
    """
    messages = [] if not system_prompt else [build_message("system", system_prompt)]
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

        if not response.text_response and response.error is not None:
            print(f"{Back.RED}{Fore.WHITE}AN ERROR OCCURRED{Style.RESET_ALL}. Terminating...\n{response.error}")
            raise response.error

        print(f"{Back.MAGENTA}{Fore.WHITE}Assistant:{Style.RESET_ALL} ", response.text_response)
        print()

        messages.append(build_message("assistant", response.text_response))
