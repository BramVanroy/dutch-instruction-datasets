from os import PathLike
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from dutch_data import AzureQuerier, Credentials
from tqdm import tqdm

from dutch_data.utils import build_message


def answer_hf_dataset(
    dataset_name: str,
    question_column: str,
    credentials_file: PathLike | str,
    credentials_profile: str,
    dout: PathLike | str,
    *,
    config_name: str | None = None,
    split: str | None = None,
    input_revision: str | None = None,
    system_column: str | None = None,
    max_num_workers: int = 1,
    timeout: float = 30.0,
    output_name: str | None = None,
    output_revision: str | None = None,
    merge_with_original: bool = True,
    verbose: bool = False,
    **kwargs,
) -> DatasetDict | None:
    """
    Translates a HuggingFace dataset using the Azure OpenAI API.
    :param dataset_name: dataset name compatible with HuggingFace datasets
    :param question_column: name of the column containing the questions to answer
    :param credentials_file: credentials file containing the Azure OpenAI API key
    :param credentials_profile: which credentials profile to use
    :param dout: output directory to save the translated dataset to. Temporary progress will also
    be saved here
    :param config_name: optional config name for the dataset
    :param split: optional split for the dataset. If not given, all splits will be translated
    :param input_revision: optional revision for the input dataset. If not given, the default revision will be used
    :param system_column: optional column containing the system messages to the questions
    :param max_num_workers: maximum number of workers to use for the querier. Note that it is no use to set a very
    high number here as the API will throttle you anyway You can try a few values to see what works best.
    :param timeout: timeout for the querier. A TimeOut error will be triggered if no response is received within
    `timeout` seconds
    :param output_name: optional hub name to push the translated dataset to. Should start with an org or username, e.g.
    "MyUserName/my-dataset-name"
    :param output_revision: optional hub branch to upload to. If not specified, will use the default branch,
    typically 'main'
    be replaced with the given source and target languages
    :param merge_with_original: whether to merge the translated dataset with the original dataset
    :param verbose: whether to print more information of the API responses
    :param kwargs: any keyword arguments to pass to the OpenAI API (such as max tokens, frequency penalty, etc.)
    :return:
    """
    # Load Azure Querier
    credentials = Credentials.from_json(credentials_file, credentials_profile)
    querier = AzureQuerier.from_credentials(credentials, max_workers=max_num_workers, timeout=timeout, verbose=verbose)

    # Load potential pre-existing data
    pdout = Path(dout).resolve()
    pdout.mkdir(parents=True, exist_ok=True)

    already_done_df = None
    pf_tmp = pdout.joinpath("tmp_openai_translations.tsv")
    if pf_tmp.exists() and pf_tmp.stat().st_size > 0:
        already_done_df = pd.read_csv(pf_tmp, sep="\t", encoding="utf-8", dtype={"idx": int})

    failed_df = None
    pf_tmp_failed = pdout.joinpath("tmp_openai_failed.tsv")
    if pf_tmp_failed.exists() and pf_tmp_failed.stat().st_size > 0:
        failed_df = pd.read_csv(pf_tmp_failed, sep="\t", encoding="utf-8", dtype={"idx": int})

    # Load dataset
    orig_dataset: DatasetDict = load_dataset(dataset_name, name=config_name, revision=input_revision)
    if split is not None:
        orig_dataset = DatasetDict({"train": orig_dataset[split]})

    columns = [question_column, system_column] if system_column else [question_column]
    orig_dataset = orig_dataset.select_columns(columns)

    # Translate
    answers = already_done_df.to_dict(orient="records") if already_done_df is not None else []
    response_colname = f"response_{credentials_profile.lower()}"
    with pf_tmp.open("a", encoding="utf-8") as fhout, pf_tmp_failed.open("a", encoding="utf-8") as fhout_failed:
        for split_name, split_dataset in orig_dataset.items():
            done_subset_idxs = set()
            if already_done_df is not None:
                done_subset_idxs = set(
                    already_done_df[
                        already_done_df["split"] == split_name
                        ]["idx"].unique()
                )
                print(
                    f"Skipping {len(done_subset_idxs)} already translated examples in {split_name}"
                )
            num_done = len(done_subset_idxs)
            if failed_df is not None:
                failed_subset_idxs = set(
                    failed_df[failed_df["split"] == split_name][
                        "idx"
                    ].unique()
                )
                print(f"Skipping {len(failed_subset_idxs)} failed examples in {split_name}")
                done_subset_idxs = done_subset_idxs.union(failed_subset_idxs)
            num_failed = len(failed_subset_idxs)

            messages = [
                (
                    sample_idx,
                    (
                        [
                            build_message("system", sample[system_column].strip()),
                            build_message("user", sample[question_column].strip()),
                        ]
                        if system_column
                        else [build_message("user", sample[question_column].strip())]
                    ),
                )
                for sample_idx, sample in enumerate(split_dataset)
                if sample_idx not in done_subset_idxs and sample.strip()
            ]

            if not messages:
                continue

            print(f"Number of messages to answer: {len(messages)}")

            for answer_result in tqdm(
                querier.query_list_of_messages(messages, return_in_order=False, **kwargs),
                total=len(messages),
                desc=f"Translating {split_name}",
            ):
                chunk = {
                    "split": split_name,
                    "column": response_colname,
                    "idx": answer_result.job_idx,
                }

                if answer_result.error is None and answer_result.result is not None:
                    chunk[response_colname] = answer_result.result.strip()
                    answers.append(chunk)
                    # Using pd for easy encoding and potential troubles with newlines and such
                    chunk_df = pd.DataFrame([chunk])
                    # Only write output header if we're at the top of the file
                    chunk_df.to_csv(fhout, index=False, header=fhout.tell() == 0, sep="\t", encoding="utf-8")
                    fhout.flush()
                    num_done += 1
                else:
                    chunk["error"] = str(answer_result.error)
                    chunk_df_failed = pd.DataFrame([chunk])
                    chunk_df_failed.to_csv(
                        fhout_failed, index=False, header=fhout_failed.tell() == 0, sep="\t", encoding="utf-8"
                    )
                    fhout_failed.flush()
                    num_failed += 1

                if verbose:
                    print(
                        f"Current progress in {split_name}: {num_done:,} done,"
                        f" {num_failed:,} failed",
                        flush=True,
                    )

    if answers:
        df = pd.DataFrame(answers)
        output_datasets = {}
        for split_name, split_group in df.groupby("split"):
            done_subset_idxs = sorted(split_group["idx"].unique())
            split_group = split_group.drop(columns=["split"])
            # Pivot so that we get the expected output format
            # Also sort by the index (rows sorted like the original dataset) and then by column name
            split_group = (
                split_group.pivot(index="idx", columns="column", values=response_colname)
                .sort_index()
                .sort_index(axis=1)
                .fillna("")
            )
            print(split_group.head(3))
            split_ds = Dataset.from_pandas(split_group, preserve_index=False)

            if merge_with_original:
                orig_split_ds = orig_dataset[split_name].select(done_subset_idxs)
                split_ds = concatenate_datasets([orig_split_ds, split_ds], axis=1)
            split_ds = split_ds.select_columns(sorted(split_ds.column_names))
            output_datasets[split_name] = split_ds

        output_datasets = DatasetDict(output_datasets)
        output_datasets.save_to_disk(pdout)

        if output_name:
            output_datasets.push_to_hub(output_name, revision=output_revision)

        return output_datasets
    else:
        return None
