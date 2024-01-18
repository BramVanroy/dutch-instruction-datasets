from typing import Annotated, Optional

import torch
import typer
from dutch_data.processor.translate import TranslationGenerator
from dutch_data.text_generator import AzureTextGenerator, HFTextGenerator, VLLMTextGenerator
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
        Optional[list[str]],
        Option(help="optional splits for the dataset. If not given, all splits will be translated"),
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
    system_prompt: Annotated[
        Optional[str],
        Option(
            help="optional system prompt to use. Should be a string with optional {src_lang} and/or {tgt_lang} fields"
            " that will be replaced with the given source and target languages. If not given, will use a default"
            " translation prompt. Can also be a dictionary with keys column names and values system prompts for"
            " that column, which is useful when you want to use different prompts for translating different"
            " columns. If None is given, will also default to the basic system prompt. 'system_prompt' can also"
            " be a file, in which case the file contents will be used as the system prompt."
        ),
    ] = None,
    columns: Annotated[
        Optional[list[str]],
        Option(
            help="optional list of column names to translate. Other columns will be dropped. If not given, all columns will be translated"
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
    Translate any dataset on the Hugging Face hub to a given language (default Dutch), optionally filtered by
    splits and columns. Depending on the given arguments, will use either a HuggingFace conversational model or the
    Azure API to translate the dataset. Will save the translated dataset to the given output directory. Optionally,
    will also upload the translated dataset to the given hub name and revision.
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

    translator = TranslationGenerator(
        text_generator=text_generator,
        dataset_name=dataset_name,
        config_name=config_name,
        splits=split,
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
        return translator.process_dataset(max_new_tokens=max_tokens, batch_size=batch_size)
    else:
        return translator.process_dataset(max_tokens=max_tokens)
