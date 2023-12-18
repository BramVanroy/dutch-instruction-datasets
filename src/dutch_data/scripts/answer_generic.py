from typing import Annotated, Optional

import typer
from dutch_data.dataset_processing import AnswerHFDataset
from dutch_data.text_generator import AzureTextGenerator, HFTextGenerator
from typer import Argument, Option


app = typer.Typer()


@app.command()
def answer(
    dataset_name: Annotated[str, Argument(help="dataset name compatible with HuggingFace datasets")],
    output_directory: Annotated[str, Argument(help="output directory to save the answered dataset to")],
    instruction_column: Annotated[
        str,
        Argument(help="column name of the dataset to answer"),
    ],
    config_name: Annotated[
        Optional[str],
        Option(help="optional config name for the dataset"),
    ] = None,
    split: Annotated[
        Optional[str],
        Option(help="optional split for the dataset. If not given, all splits will be answered"),
    ] = None,
    revision: Annotated[
        Optional[str],
        Option(help="optional revision for the dataset. If not given, will load the main revision"),
    ] = None,
    hf_model_name: Annotated[
        Optional[str],
        Option(
            help="HuggingFace model name to use for the text generator. Note that currently only conversational style models are supported"
        ),
    ] = None,
    credentials_file: Annotated[Optional[str], Option(help="JSON file containing credentials")] = None,
    credentials_profiles: Annotated[
        Optional[list[str]],
        Option(
            "-p",
            "--credentials_profiles",
            help="which credential profile(s) (key) to use from the credentials file. If not given, will use all"
            " profiles in a cyclical manner to optimize API calls",
        ),
    ] = None,
    response_column: Annotated[
        str,
        Option(help="column name where to write the responses to"),
    ] = "response",
    output_hub_name: Annotated[
        Optional[str],
        Option(
            help="optional hub name to push the answered dataset to. Should start with an org or username,"
            " e.g. 'MyUserName/my-dataset-name'"
        ),
    ] = None,
    output_hub_revision: Annotated[
        Optional[str],
        Option(help="hub branch to upload to. If not specified, will use the default branch, typically 'main'."),
    ] = None,
    max_workers: Annotated[
        int, Option("--max-workers", "-j", help="(azure) how many parallel workers to use to query the API")
    ] = 6,
    max_retries: Annotated[int, Option(help="(azure) how many times to retry on errors")] = 3,
    max_tokens: Annotated[int, Option(help="max new tokens to generate")] = 2048,
    timeout: Annotated[float, Option("--timeout", "-t", help="(azure) timeout in seconds for each API call")] = 30.0,
    verbose: Annotated[
        bool, Option("--verbose", "-v", help="(azure) whether to print more information of the API responses")
    ] = False,
):
    """
    Answer a column of any dataset on the Hugging Face hub, optionally filtered by split and columns. Depending on the
    given arguments, will use either a HuggingFace conversational model or the Azure API to answer the dataset.
    Will save the answered dataset to the given output directory. Optionally, will also upload the
    dataset to the given hub name and revision.
    """
    if hf_model_name is None and credentials_file is None:
        raise ValueError("Either hf_model_name or credentials_file must be given")

    if hf_model_name:
        text_generator = HFTextGenerator(hf_model_name)
    else:
        text_generator = AzureTextGenerator.from_json(
            credentials_file,
            credentials_profiles,
            max_workers=max_workers,
            timeout=timeout,
            max_retries=max_retries,
            verbose=verbose,
        )

    answerer = AnswerHFDataset(
        text_generator=text_generator,
        dataset_name=dataset_name,
        content_role_columns=instruction_column,
        config_name=config_name,
        split=split,
        revision=revision,
        response_column=response_column,
        dout=output_directory,
        output_hub_name=output_hub_name,
        output_hub_revision=output_hub_revision,
        merge_with_original=True,
        verbose=verbose,
    )

    if hf_model_name:
        return answerer.process_dataset(max_new_tokens=max_tokens)
    else:
        return answerer.process_dataset(max_tokens=max_tokens)
