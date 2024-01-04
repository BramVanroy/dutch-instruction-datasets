import json
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TextIO

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from dutch_data.text_generator import TextGenerator


@dataclass
class BaseHFDatasetProcessor(ABC):
    """
    Base class for processing HuggingFace datasets.
    :param dataset_name: dataset name compatible with HuggingFace datasets
    :param text_generator: text generator to use for querying. This can be a HuggingFace pipeline, an Azure pipeline,
    or any other TextGenerator subclass that implements the `query_messages` method
    :param dout: output directory to save the translated dataset to. Temporary progress will also be saved here
    :param config_name: optional config name for the dataset
    :param splits: optional split or list of splits for the dataset. If not given, all splits will be translated
    :param revision: optional revision for the dataset. If not given, will load the main revision
    :param max_samples: maximum number of samples to translate. Useful for testing
    :param output_hub_name: optional hub name to push the translated dataset to. Should start with an org or username,
     e.g. "MyUserName/my-dataset-name"
    :param output_hub_revision: optional hub branch to upload to. If not specified, will use the default branch,
    typically 'main' be replaced with the given source and target languages
    :param merge_with_original: whether to merge the translated dataset with the original dataset
    :param verbose: whether to print more information
    """

    dataset_name: str
    text_generator: TextGenerator
    dout: PathLike | str
    config_name: str | None = None
    splits: list[str] | str | None = None
    revision: str | None = None
    max_samples: int | None = None
    output_hub_name: str | None = None
    output_hub_revision: str | None = None
    merge_with_original: bool = True
    verbose: bool = False

    def __post_init__(self):
        self.dout: Path = Path(self.dout).resolve()
        self.dout.mkdir(parents=True, exist_ok=True)
        if self.splits is not None and isinstance(self.splits, str):
            self.splits = [self.splits]

    def _load_done_failed(self) -> tuple[Path, list[dict] | None, Path, list[dict] | None]:
        """
        Load the already done and failed dataframes from disk if they exist and are not empty
        :return: a tuple containing the path to the temporary file of results, the already done dataframe, the path
        to the temporary file of failed results, and the dataframe with failed results
        """
        # Backwards-compatible to when we used TSV files
        # Done
        pf_tmp_old = self.dout.joinpath("tmp_openai_done.tsv")
        pf_tmp = self.dout.joinpath("tmp_openai_done.jsonl")

        if not pf_tmp.exists() and pf_tmp_old.exists() and pf_tmp_old.stat().st_size > 0:
            already_done_df = pd.read_csv(pf_tmp_old, sep="\t", encoding="utf-8", dtype={"idx": int})
            already_done_df.to_json(pf_tmp, orient="records", lines=True)

        done_samples = None
        if pf_tmp.exists() and pf_tmp.stat().st_size > 0:
            done_samples = [json.loads(line) for line in pf_tmp.read_text(encoding="utf-8").splitlines()]

        # Failed
        pf_tmp_failed_old = self.dout.joinpath("tmp_openai_failed.tsv")
        pf_tmp_failed = self.dout.joinpath("tmp_openai_failed.jsonl")

        if not pf_tmp_failed.exists() and pf_tmp_failed_old.exists() and pf_tmp_failed_old.stat().st_size > 0:
            failed_df = pd.read_csv(pf_tmp_failed_old, sep="\t", encoding="utf-8", dtype={"idx": int})
            failed_df.to_json(pf_tmp_failed, orient="records", lines=True)

        failed_samples = None
        if pf_tmp_failed.exists() and pf_tmp_failed.stat().st_size > 0:
            failed_samples = [json.loads(line) for line in pf_tmp_failed.read_text(encoding="utf-8").splitlines()]

        return pf_tmp, done_samples, pf_tmp_failed, failed_samples

    @staticmethod
    def _get_done_failed_subset_idxs(
        done_samples: list[dict], failed_samples: list[dict], split_filter: str, column_filter: str | None = None
    ) -> tuple[set[int], int, set[int], int]:
        """
        Load the indices of the examples that have already been processed by the API or failed.
        :param done_samples: list of done samples
        :param failed_samples: list of failed samples
        :param split_filter: split to filter on (typically "train", "validation" or "test")
        :param column_filter: an optional additional filter on the column name
        :return: a tuple containing the indices of the already done examples, the length of that set, the indices of
         the failed examples, and the length of that set
        """

        def filter_samples(samples: list[dict]):
            return [
                item
                for item in samples
                if item["split"] == split_filter and (not column_filter or item["column"] == column_filter)
            ]

        done_subset_idxs = set()
        if done_samples:
            done_subset_idxs = set([item["idx"] for item in filter_samples(done_samples)])

        failed_subset_idxs = set()
        if failed_samples is not None:
            failed_subset_idxs = set([item["idx"] for item in filter_samples(failed_samples)])
            done_subset_idxs = done_subset_idxs.union(failed_subset_idxs)

        return done_subset_idxs, len(done_subset_idxs), failed_subset_idxs, len(failed_subset_idxs)

    def _load_dataset(self) -> DatasetDict:
        """
        Load the dataset from Hugging Face datasets. Optionally restricted to a specific split.
        :return: a loaded DatasetDict
        """
        orig_dataset: DatasetDict = load_dataset(self.dataset_name, name=self.config_name, revision=self.revision)

        if self.splits:
            for splitname in self.splits:
                if splitname not in orig_dataset:
                    raise ValueError(
                        f"Split {splitname} not found in dataset {self.dataset_name}."
                        f" Available splits: {orig_dataset.keys()}"
                    )
            orig_dataset = DatasetDict({k: v for k, v in orig_dataset.items() if k in self.splits})

        if not orig_dataset:
            raise ValueError(
                "Dataset appears to empty, perhaps because you selected an empty dataset:"
                f" dataset_name={self.dataset_name}, config_name={self.config_name},"
                f" revision={self.revision}, splits={self.splits}"
            )

        return orig_dataset

    @staticmethod
    def _write_row_to_fh(fh: TextIO, row: dict):
        """
        Write a row to a file handle as a JSON string.
        :param fh: file handle
        :param row: row to write, in dictionary format
        """
        fh.write(f"{json.dumps(row)}\n")
        fh.flush()

    @abstractmethod
    def process_dataset(self, **kwargs):
        """
        Main entry point for processing the dataset. This should load the dataset, prepare the messages,
        send them to the API and postprocess the results.
        """
        pass

    def _postprocess_dataset(self, results: list[dict], orig_dataset: DatasetDict, pivot_column: str) -> DatasetDict:
        """
        Postprocess the results from the API and save the translated dataset to disk. Optionally upload it to the
        Hugging Face hub.
        :param results: a list of dictionaries, containing the processed rows
        :param orig_dataset: an original dataset dict to potentially merge with (depending on self.merge_with_original)
        :param pivot_column: the column to pivot the resulting rows on
        :return: a DatasetDict with the result, optionally merged with the original
        """
        df = pd.DataFrame(results)
        output_datasets = {}
        for split_name, split_group in df.groupby("split"):
            done_subset_idxs = sorted(split_group["idx"].unique())
            split_group = split_group.drop(columns=["split"])
            # Pivot so that we get the expected output format
            # Also sort by the index (rows sorted like the original dataset) and then by column name
            split_group = (
                split_group.pivot(index="idx", columns="column", values=pivot_column)
                .sort_index()
                .sort_index(axis=1)
                .fillna("")
            )
            if self.verbose:
                print("Unmerged results:")
                print(split_group.head(3))
            split_ds = Dataset.from_pandas(split_group, preserve_index=False)

            if self.merge_with_original:
                orig_split_ds = orig_dataset[split_name].select(done_subset_idxs)
                split_ds = concatenate_datasets([orig_split_ds, split_ds], axis=1)
            split_ds = split_ds.select_columns(sorted(split_ds.column_names))
            output_datasets[split_name] = split_ds

        output_datasets = DatasetDict(output_datasets)
        output_datasets.save_to_disk(self.dout)

        if self.output_hub_name:
            output_datasets.push_to_hub(self.output_hub_name, revision=self.output_hub_revision)

        return output_datasets

    @abstractmethod
    def _prepare_messages(self, *args, **kwargs) -> list[tuple[int, list[dict[[str, str]]]]]:
        """
        Prepare the messages to send to the API. This should return a list of tuples, where each tuple contains
        the job index and a list of messages to send to the API (which are dictionaries).
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @staticmethod
    def _failed_items_check(pf_tmp_failed: Path):
        if pf_tmp_failed.exists() and pf_tmp_failed.stat().st_size > 0:
            failed_text = pd.read_json(pf_tmp_failed, encoding="utf-8", orient="records", lines=True)

            error_counter = Counter()
            for error in failed_text["error"]:
                error = error.lower()
                if "error code: 429" in error:
                    error_counter["rate_limit"] += 1
                elif "empty conversation" in error:
                    error_counter["empty"] += 1
                elif "timed out" in error:
                    error_counter["time_out"] += 1

            num_errors = error_counter.total()
            if num_errors:
                print(
                    f"Warning: your 'failed' file contains {num_errors:,} failed items that were caused by API rate or"
                    f" quota limits. You may wish to remove those from the 'failed' file in the output directory,"
                    f" and try again.\n"
                    f"('error code: 429': {error_counter['rate_limit']};"
                    f" 'empty conversation': {error_counter['empty']};"
                    f" 'timed out': {error_counter['time_out']})",
                    flush=True,
                    file=sys.stderr,
                )
