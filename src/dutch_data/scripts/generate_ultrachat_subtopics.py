import json
import re
from pathlib import Path
from typing import Annotated, Optional

import typer
from dutch_data import AzureQuerier, Credentials
from dutch_data.ultrachat.conceptual_fields import WORLD_CONCEPTS
from tqdm import tqdm
from typer import Argument, Option


_SUBTOPIC_PROMPT = (
    "Geef een gevarieerde lijst van {num_topics} onderwerpen die te maken hebben met, of subdomeinen"
    " zijn van, '{concept}'. Geef geen andere informatie en lijst deze onderwerpen op gescheiden door"
    " een komma."
)

app = typer.Typer()


def build_message(concept: str, num_topics: int = 30) -> list[dict[str, str]]:
    """
    Build a message for the API to generate subtopics for a world concept.
    :param concept: the concept to generate subtopics for
    :param num_topics: the number of subtopics to generate
    :return: a list of messages, in this case only containing one message (a dictionary with a role and content key)
    """
    return [
        {
            "role": "user",
            "content": _SUBTOPIC_PROMPT.format(num_topics=num_topics, concept=concept),
        }
    ]


@app.command()
def generate_subtopics(
    credentials_file: Annotated[str, typer.Argument(help="JSON file containing credentials")],
    credentials_profile: Annotated[
        str, Argument(help="which credential profile (key) to use from the credentials file")
    ],
    *,
    output_file: Annotated[Optional[str], Option(help="output JSON file to write results to")] = None,
    num_topics: Annotated[int, Option(help="how many subtopics to generate for each world concept")] = 30,
    max_num_workers: Annotated[int, Option(help="how many parallel workers to use to query the API")] = 6,
) -> dict[str, list[str]]:
    """
    Generate subtopics for the world concepts and optionally write them to a JSON file.
    """
    credentials = json.loads(Path(credentials_file).read_text(encoding="utf-8"))
    try:
        credentials = credentials[credentials_profile]
    except KeyError:
        raise KeyError(f"Credentials profile {credentials_profile} not found in credentials file")

    credentials = Credentials(**credentials)
    querier = AzureQuerier.from_credentials(credentials, max_workers=max_num_workers)

    list_of_messages = [build_message(concept, num_topics=num_topics) for concept in WORLD_CONCEPTS]

    data = {}
    for concept, response in tqdm(
        zip(WORLD_CONCEPTS, querier.query_list_of_messages(list_of_messages=list_of_messages, return_in_order=True)),
        total=len(WORLD_CONCEPTS),
    ):
        topics = sorted([re.sub(r"^\W*(.*?)\W*$", r"\1", topic).strip() for topic in response.lower().split(",")])
        data[concept] = topics
        if output_file is None:
            print(concept.upper())
            print(topics)
            print()

    if output_file is not None:
        Path(output_file).write_text(json.dumps(data, indent=4), encoding="utf-8")

    return data
