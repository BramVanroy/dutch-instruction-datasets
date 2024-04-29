from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from dutch_data.processor.base import DatasetGenerator
from tqdm import tqdm


@dataclass
class AnswerGenerator(DatasetGenerator):
    """
    Answers a HuggingFace dataset using the Azure OpenAI API.
    :param user_column: which column contains the user's message
    :param system_column: (optional) which column contains the system's message
    :param system_message: (optional) a system message to use for all examples. If a file path is given, the content
    of the file will be used as the system message
    :param response_column: which column to write answers to
    """

    user_column: str | None = None
    system_column: str | None = None
    system_message: str | PathLike | None = None
    response_column: str = "response"

    def __post_init__(self):
        super().__post_init__()
        if self.user_column is None:
            raise ValueError("You must pass a column that contains the user messages.")

        if self.system_column and self.system_message:
            raise ValueError("You cannot pass both a system column and a system message.")

        if self.system_message:
            try:
                pfsys = Path(self.system_message)
                if pfsys.exists() and pfsys.is_file():
                    self.system_message = pfsys.read_text(encoding="utf-8")
            except Exception:
                # Can error for very long strings
                pass

    def process_dataset(self, only_save_current: bool = False, **kwargs):
        orig_dataset = self._load_dataset()

        for split, subset in orig_dataset.items():
            if self.response_column in subset.column_names:
                raise ValueError(
                    f"Dataset already contains a column called '{self.response_column}'. Please choose another name."
                )

            if self.user_column not in subset.column_names or (
                self.system_column and self.system_column not in subset.column_names
            ):
                raise ValueError(f"Dataset ({split} split) does not contain the user and/or system columns.")

        pf_tmp, done_samples, pf_tmp_failed, failed_samples = self._load_done_failed()

        if not only_save_current:
            with pf_tmp.open("a", encoding="utf-8") as fhout, pf_tmp_failed.open(
                "a", encoding="utf-8"
            ) as fhout_failed:
                for split_name, split_dataset in orig_dataset.items():
                    if self.max_samples is not None:
                        split_dataset = split_dataset.select(range(self.max_samples))

                    # Get the IDs of the examples that have already been translated or failed
                    done_subset_idxs, num_done, failed_subset_idxs, num_failed = self._get_done_failed_subset_idxs(
                        done_samples, failed_samples, split_name
                    )

                    print(
                        f"Skipping {len(done_subset_idxs)} already answered examples in {split_name}"
                        f"\nSkipping {len(failed_subset_idxs)} failed examples in {split_name}"
                    )

                    messages = self._prepare_messages(split_dataset, done_subset_idxs)

                    if not messages:
                        continue

                    print(f"Number of messages to answer: {len(messages)}")

                    for answer_response in (
                        pbar := tqdm(
                            self.text_generator.batch_query_messages(messages, **kwargs),
                            total=len(messages),
                        )
                    ):
                        result_row = {
                            "split": split_name,
                            "column": self.response_column,
                            "idx": answer_response.job_idx,
                        }

                        if answer_response.error is None and answer_response.text_response is not None:
                            result_row[self.response_column] = answer_response.text_response.strip()
                            done_samples.append(result_row)
                            self._write_row_to_fh(fhout, result_row)
                            num_done += 1
                        else:
                            result_row["error"] = str(answer_response.error)
                            self._write_row_to_fh(fhout_failed, result_row)
                            num_failed += 1

                        pbar.set_description(f"{split_name} ({num_done:,} ✓ | {num_failed:,} ✗)")

        self._failed_items_check(pf_tmp_failed)
        if done_samples:
            output_datasets = self._postprocess_dataset(done_samples, orig_dataset, self.response_column)
            return output_datasets
        else:
            return None

    def _prepare_messages(self, dataset, done_subset_idxs):
        def build_message(sample, sample_idx):
            if self.system_column:
                return (
                    sample_idx,
                    [
                        {"role": "system", "content": sample[self.system_column]},
                        {"role": "user", "content": sample[self.user_column]},
                    ],
                )
            elif self.system_message:
                return (
                    sample_idx,
                    [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": sample[self.user_column]},
                    ],
                )
            else:
                return sample_idx, [{"role": "user", "content": sample[self.user_column]}]

        messages = [
            build_message(sample, sample_idx)
            for sample_idx, sample in enumerate(dataset)
            if sample_idx not in done_subset_idxs
        ]

        return messages
