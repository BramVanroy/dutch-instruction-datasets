from typing import Annotated, Optional

import typer
from dutch_data.translate_hf import SYSTEM_TRANSLATION_PROMPT, translate_hf_dataset
from typer import Argument, Option


app = typer.Typer()


@app.command()
def translate_orcadpo_system_question(
    credentials_file: Annotated[str, typer.Argument(help="JSON file containing credentials")],
    credentials_profile: Annotated[
        str, Argument(help="which credential profile (key) to use from the credentials file")
    ],
    input_hub_name: Annotated[
        str,
        Argument(
            help="Hub dataset name to use as input. Should start with an org or username,"
            " e.g. 'MyUserName/my-dataset-name'"
        ),
    ],
    output_directory: Annotated[str, Argument(help="output directory to save the translated dataset to")],
    *,
    system_msg_column: Annotated[str, Option(help="column name for system message")] = "system",
    question_column: Annotated[str, Option(help="column name for question")] = "question",
    input_hub_revision: Annotated[
        Optional[str],
        Option(help="hub branch to use for input. If not specified, will use the default branch, typically 'main'."),
    ] = None,
    output_hub_name: Annotated[
        Optional[str],
        Option(
            help="optional hub name to push the translated dataset to. Should start with an org or username,"
            " e.g. 'MyUserName/my-dataset-name'"
        ),
    ] = None,
    output_hub_revision: Annotated[
        Optional[str],
        Option(help="hub branch to upload to. If not specified, will use the default branch, typically 'main'."),
    ] = None,
    max_tokens: Annotated[int, Option(help="max new tokens to generate")] = 2048,
    max_num_workers: Annotated[
        int, Option(help="how many parallel workers to use to query the API. Only used when using OpenAI's API")
    ] = 6,
    timeout: Annotated[
        float, Option(help="timeout in seconds for each API call. Only used when using OpenAI's API")
    ] = 60.0,
):
    return translate_hf_dataset(
        dataset_name="Intel/orca_dpo_pairs",
        tgt_lang=tgt_lang,
        credentials_file=credentials_file,
        credentials_profile=credentials_profile,
        dout=output_directory,
        columns=["system", "question"],
        hub_name=hub_name,
        hub_revision=hub_revision,
        merge_with_original=True,
        max_num_workers=max_num_workers,
        system_prompt_template=sys_msg,
        max_tokens=max_tokens,
        timeout=timeout,
    )
