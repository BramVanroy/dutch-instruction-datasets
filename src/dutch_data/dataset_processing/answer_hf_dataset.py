from dataclasses import dataclass

from dutch_data.dataset_processing.base_processor import BaseHFDatasetProcessor
from tqdm import tqdm


@dataclass
class AnswerHFDataset(BaseHFDatasetProcessor):
    """
    Answers a HuggingFace dataset using the Azure OpenAI API.
    :param content_role_columns: names of the columns and their associated roles in the conversation in order of
    appearance. For example, if you have a column called "system" that contains the system messages and a column
    called "user" that contains the user messages, you would pass {"system": "system", "user": "user"}. If you have
    a column called "question" that contains the questions and no system messages, then you would pass
    {"question": "user"}.
    :param response_column: which column to write answers to
    """

    content_role_columns: dict[str, str] = None
    response_column: str = "response"

    def __post_init__(self):
        if self.content_role_columns is None:
            raise ValueError(
                "You must pass a dictionary of content role columns to 'content_role_columns', which indicates"
                "which columns to include as well as their role (e.g. 'system', 'user' or 'assistant')."
            )
        elif any(role not in ("assistant", "system", "user") for role in self.content_role_columns.values()):
            raise ValueError(
                "When passing a dictionary as 'content_role_columns', its keys must be column names in the dataset"
                " and its values must be one of 'assistant', 'system' or 'user'."
            )

    def process_dataset(self, **kwargs):
        orig_dataset = self._load_dataset()

        if self.response_column in orig_dataset.column_names:
            raise ValueError(
                f"Dataset already contains a column called '{self.response_column}'. Please choose another name."
            )

        if any(colname not in orig_dataset.column_names for colname in self.content_role_columns.keys()):
            raise ValueError(
                f"Dataset does not contain all the columns in 'content_role_columns':"
                f" {self.content_role_columns.keys()}"
            )

        pf_tmp, already_done_df, pf_tmp_failed, failed_df = self._load_done_failed_dfs()

        answers = already_done_df.to_dict(orient="records") if already_done_df is not None else []

        with pf_tmp.open("a", encoding="utf-8") as fhout, pf_tmp_failed.open("a", encoding="utf-8") as fhout_failed:
            for split_name, split_dataset in orig_dataset.items():
                if self.max_samples is not None:
                    split_dataset = split_dataset.select(range(self.max_samples))

                # Get the IDs of the examples that have already been translated or failed
                done_subset_idxs, num_done, failed_subset_idxs, num_failed = self._get_done_failed_subset_idxs(
                    already_done_df, failed_df, split_name
                )

                print(
                    f"Skipping {len(done_subset_idxs)} already answered examples in {split_name}"
                    f"\nSkipping {len(failed_subset_idxs)} failed examples in {split_name}"
                )

                messages = self._prepare_messages(split_dataset, done_subset_idxs)

                if not messages:
                    continue

                print(f"Number of messages to answer: {len(messages)}")

                for answer_result in tqdm(
                    self.text_generator.batch_query_messages(messages, return_in_order=False, **kwargs),
                    total=len(messages),
                    desc=f"Answering {split_name}",
                ):
                    result_row = {
                        "split": split_name,
                        "column": self.response_column,
                        "idx": answer_result.job_idx,
                    }

                    if answer_result.error is None and answer_result.text_response is not None:
                        result_row[self.response_column] = answer_result.text_response.strip()
                        answers.append(result_row)
                        self._write_row_to_fh(fhout, result_row)
                        num_done += 1
                    else:
                        result_row["error"] = str(answer_result.error)
                        self._write_row_to_fh(fhout_failed, result_row)
                        num_failed += 1

        if answers:
            output_datasets = self._postprocess_dataset(answers, orig_dataset, self.response_column)
            return output_datasets
        else:
            return None

    def _prepare_messages(self, dataset, done_subset_idxs):
        messages = [
            (
                sample_idx,
                [{"role": role, "content": sample[colname]} for colname, role in self.content_role_columns.items()],
            )
            for sample_idx, sample in enumerate(dataset)
            if sample_idx not in done_subset_idxs
        ]

        return messages
