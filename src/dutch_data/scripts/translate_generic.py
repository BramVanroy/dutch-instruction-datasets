from typing import Annotated, Optional

import typer
from dutch_data.dataset_processing.translate_hf_dataset import TranslateHFDataset
from dutch_data.text_generator import AzureTextGenerator, HFTextGenerator
from typer import Argument, Option


app = typer.Typer()


@app.command()
def translate(
    dataset_name: Annotated[str, Argument(help="dataset name compatible with HuggingFace datasets")],
    output_directory: Annotated[str, Argument(help="output directory to save the translated dataset to")],
    config_name: Annotated[
        Optional[str],
        Option(help="optional config name for the dataset"),
    ] = None,
    split: Annotated[
        Optional[str],
        Option(help="optional split for the dataset. If not given, all splits will be translated"),
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
    src_lang: Annotated[
        Optional[str],
        Option(
            help="source language to translate from. Will be used in the default system prompt or your custom prompt, see 'system_prompt'"
        ),
    ] = None,
    tgt_lang: Annotated[
        Optional[str],
        Option(
            help="target language to translate to. Will be used in the default system prompt or your custom prompt, see 'system_prompt'"
        ),
    ] = None,
    columns: Annotated[
        Optional[list[str]],
        Option(
            help="optional list of column names to translate. Other columns will be dropped. If not given, all columns will be translated"
        ),
    ] = None,
    system_prompt: Annotated[
        Optional[str],
        Option(
            help="optional system prompt to use for the translation to tell the model it has to translate."
            " If not given, will use a default prompt. Note that that will require you to specify a src_lang and tgt_lang"
        ),
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
    Translate any dataset on the Hugging Face hub to a given language (default Dutch), optionally filtered by
    split and columns. Depending on the given arguments, will use either a HuggingFace conversational model or the
    Azure API to translate the dataset. Will save the translated dataset to the given output directory. Optionally,
    will also upload the translated dataset to the given hub name and revision.
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

    translator = TranslateHFDataset(
        text_generator=text_generator,
        dataset_name=dataset_name,
        config_name=config_name,
        split=split,
        revision=revision,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        dout=output_directory,
        columns=columns,
        output_hub_name=output_hub_name,
        output_hub_revision=output_hub_revision,
        merge_with_original=True,
        system_prompt=system_prompt,
        verbose=verbose,
    )

    if hf_model_name:
        return translator.process_dataset(max_new_tokens=max_tokens)
    else:
        return translator.process_dataset(max_tokens=max_tokens)
