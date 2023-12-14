from typing import Annotated, Optional

import typer
from dutch_data.dataset_processing import answer_hf_dataset
from typer import Argument, Option


app = typer.Typer()


@app.command()
def answer_orcadpo_azure(
    credentials_file: Annotated[str, typer.Argument(help="JSON file containing credentials")],
    credentials_profile: Annotated[
        str, Argument(help="which credential profile (key) to use from the credentials file")
    ],
    input_hub_name: Annotated[
        str,
        Argument(
            help="hub name to get a dataset from. Should start with an org or username,"
            " e.g. 'MyUserName/my-dataset-name'"
        ),
    ],
    question_column: Annotated[
        str,
        Argument(help="name of the column containing the questions to answer"),
    ],
    output_directory: Annotated[str, Argument(help="output directory to save the answered dataset to")],
    *,
    system_column: Annotated[
        Optional[str],
        Option(help="optional column containing the system messages to the questions"),
    ] = None,
    input_hub_revision: Annotated[
        Optional[str],
        Option(help="hub branch to retrieve. If not specified, will use the default branch, typically 'main'."),
    ] = None,
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
    max_num_workers: Annotated[
        int, Option("--max-num-workers", "-j", help="how many parallel workers to use to query the API")
    ] = 6,
    max_tokens: Annotated[int, Option(help="max new tokens to generate with the API")] = 2048,
    timeout: Annotated[float, Option("--timeout", "-t", help="timeout in seconds for each API call")] = 30.0,
    verbose: Annotated[
        bool, Option("--verbose", "-v", help="whether to print more information of the API responses")
    ] = False,
):
    """
    Let Azure answer the questions in a dataset from the Orca DPO set. (But should also work with other datasets
    """
    return answer_hf_dataset(
        dataset_name=input_hub_name,
        question_column=question_column,
        input_revision=input_hub_revision,
        credentials_file=credentials_file,
        credentials_profile=credentials_profile,
        dout=output_directory,
        system_column=system_column,
        output_name=output_hub_name,
        output_revision=output_hub_revision,
        merge_with_original=True,
        max_num_workers=max_num_workers,
        max_tokens=max_tokens,
        timeout=timeout,
        verbose=verbose,
    )
