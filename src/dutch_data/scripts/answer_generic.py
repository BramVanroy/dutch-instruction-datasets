from typing import Annotated, Optional

import torch
import typer
from dutch_data.processor.answer import AnswerGenerator
from dutch_data.text_generator import AzureTextGenerator, HFTextGenerator, VLLMTextGenerator
from typer import Argument, Option


app = typer.Typer()


@app.command()
def answer(
    dataset_name: Annotated[str, Argument(help="dataset name compatible with HuggingFace datasets")],
    output_directory: Annotated[str, Argument(help="output directory to save the answered dataset to")],
    user_column: Annotated[
        str,
        Argument(help="column name of the dataset to answer"),
    ],
    config_name: Annotated[
        Optional[str],
        Option(help="optional config name for the dataset"),
    ] = None,
    split: Annotated[
        Optional[list[str]],
        Option(help="optional splits for the dataset. If not given, all splits will be answered"),
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
    use_vllm: Annotated[
        bool,
        typer.Option(
            help="whether to use VLLM for faster inference on Hugging Face models. Note that this will use and start"
            " up VLLM from within Python and is therefore different from 'vllm_endpoint', which expects an already"
            " running VLLM inference server"
        ),
    ] = False,
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
    system_column: Annotated[
        Optional[str],
        Option(help="optional system column that will be included as a system message"),
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
    max_workers: Annotated[
        int, Option("--max-workers", "-j", help="(azure) how many parallel workers to use to query the API")
    ] = 1,
    max_retries: Annotated[int, Option(help="(azure) how many times to retry on errors")] = 3,
    max_tokens: Annotated[int, Option(help="max new tokens to generate")] = 2048,
    timeout: Annotated[float, Option("--timeout", "-t", help="(azure) timeout in seconds for each API call")] = 30.0,
    verbose: Annotated[
        bool, Option("--verbose", "-v", help="(azure) whether to print more information of the API responses")
    ] = False,
    device_map: Annotated[
        Optional[str],
        Option(help="(hf) device map to use for the model. Can be 'auto' or a device ID (e.g. 'cuda:0')"),
    ] = None,
    load_in_8bit: Annotated[
        bool, Option(help="(hf) whether to load the model in 8bit precision to save memory")
    ] = False,
    load_in_4bit: Annotated[
        bool, Option(help="(hf) whether to load the model in 4bit precision to save memory")
    ] = False,
    torch_dtype: Annotated[
        Optional[str],
        Option(help="(hf) data type to use for the model, e.g. 'bfloat16' or 'auto'"),
    ] = None,
    batch_size: Annotated[
        int,
        Option(
            help="(hf/use_vllm) batch size for inference. Note that higher values not necessarily increase speed! When"
            " 'use_vllm', this is just a pseudo-batch. VLLM does batching by itself but to get responses of"
            " smaller batches to write them to output files more quickly (rather than at the very end only) we can"
            " set a smaller pseudo-batch size. In the VLLM case, I recommend to set this to a non-1 value."
        ),
    ] = 1,
):
    """
    Answer a column of any dataset on the Hugging Face hub, optionally filtered by split and columns. Depending on the
    given arguments, will use either a HuggingFace conversational model or the Azure API to answer the dataset.
    Will save the answered dataset to the given output directory. Optionally, will also upload the
    dataset to the given hub name and revision.
    """
    if hf_model_name is None and credentials_file is None:
        raise ValueError("Either hf_model_name or credentials_file must be given")

    if use_vllm and not hf_model_name:
        raise ValueError("When using 'use_vllm', a model name 'hf_model_name' must be specified")

    if hf_model_name:
        if use_vllm:
            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
            text_generator = VLLMTextGenerator(model_name=hf_model_name, tensor_parallel_size=num_devices)
        else:
            text_generator = HFTextGenerator(
                hf_model_name,
                device_map=device_map,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                torch_dtype=torch_dtype,
            )
    else:
        text_generator = AzureTextGenerator.from_json(
            credentials_file,
            credentials_profiles,
            max_workers=max_workers,
            timeout=timeout,
            max_retries=max_retries,
            verbose=verbose,
        )

    answerer = AnswerGenerator(
        text_generator=text_generator,
        dataset_name=dataset_name,
        user_column=user_column,
        system_column=system_column,
        config_name=config_name,
        splits=split,
        revision=revision,
        response_column=response_column,
        dout=output_directory,
        output_hub_name=output_hub_name,
        output_hub_revision=output_hub_revision,
        merge_with_original=True,
        verbose=verbose,
    )

    if hf_model_name:
        return answerer.process_dataset(max_new_tokens=max_tokens, batch_size=batch_size)
    else:
        return answerer.process_dataset(max_tokens=max_tokens)
