import json
from dataclasses import dataclass
from pathlib import Path
from random import choice

from dutch_data import AzureTextGenerator
from dutch_data.azure_utils.utils import extract_conversation_from_json, extract_conversation_from_string
from dutch_data.dataset_processing.base_processor import BaseHFDatasetProcessor
from tqdm import tqdm


@dataclass
class ConversationHFDataset(BaseHFDatasetProcessor):
    """
    Build a conversation from a HuggingFace dataset.
    :param seed_column: column name of the dataset to use as seed question
    :param system_prompt: system prompt that has the {subject} field to fill in with the seed question
    as well as an optional {persona} field to fill in with a random persona from the personas dict. NOTE:
    the system prompt must request the expected format, e.g., the system prompt could include an example like so:

        Example output:

        ```
        user: [first user message]
        assistant: [reply to the first user message

        user: [second user message]
        assistant: [reply to the second user message]
        ```

    We'll try to manually extract all "user": "...", "assistant": "..." pairs from the model response so make sure to
    correctly specify the user_id and assistant_id fields below (in the example above that would 'user: ' and
    'assistant: '). If you want to use a persona, make sure to include the {persona} field in the system prompt.

    If the given system_prompt is a file, we will read its contents.
    :param user_id: id description of the user. Make sure to include colons or other characters that are
    part of the identifier. E.g., 'user: '. Must be part of system prompt.
    :param assistant_id: id description of the assistant. Make sure to include colons or other characters that are
    part of the identifier. E.g., 'assistant: '. Must be part of system prompt.
    :param personas: optional personas to use with the system_prompt. If a string, expects a json file
    with a dictionary of personas. If a dictionary, expects a dictionary of personas.
    :param output_column: column name to save the conversation to
    """

    seed_column: str | None = None
    system_prompt: str | None = None
    user_id: str | None = "user: "
    assistant_id: str | None = "assistant: "
    personas: dict[str, str] | str | None = None
    output_column: str = "messages"

    def __post_init__(self):
        super().__post_init__()

        if self.seed_column is None:
            raise ValueError("You must pass a column that contains the seed question.")

        if self.personas is None:
            if "{persona}" in self.system_prompt:
                raise ValueError(
                    "You must pass a dictionary of personas or a JSON file with personas if you want to use the"
                    " {persona} field in the system prompt."
                )
        elif isinstance(self.personas, str):
            pfpersonas = Path(self.personas)
            if not pfpersonas.is_file():
                raise ValueError(
                    "If 'personas' is a string, it must point to a JSON file with persona names as keys and persona"
                    " properties as descriptions."
                )
            self.personas = json.loads(pfpersonas.read_text(encoding="utf-8"))

        if self.system_prompt is None:
            raise ValueError("You must pass a system prompt.")

        pfsys = Path(self.system_prompt)
        if pfsys.is_file():
            self.system_prompt = pfsys.read_text(encoding="utf-8")

        if not self.user_id or self.user_id not in self.system_prompt or not self.assistant_id or self.assistant_id not in self.system_prompt:
            raise ValueError(
                "You must pass a user_id and assistant_id that is also given in an example output in your"
                " system_prompt. This is necessary so that the model knows how to structure the output in terms of"
                " the user and assistant roles."
            )

    def process_dataset(self, **kwargs):
        orig_dataset = self._load_dataset()
        if self.output_column in orig_dataset.column_names:
            raise ValueError(
                f"Dataset already contains a column called '{self.output_column}'. Please choose another name."
            )

        for split, subset in orig_dataset.items():
            if self.seed_column not in subset.column_names:
                raise ValueError(f"Dataset ({split} split) does not contain the seed column.")

        pf_tmp, already_done_df, pf_tmp_failed, failed_df = self._load_done_failed_dfs()

        convos = already_done_df.to_dict(orient="records") if already_done_df is not None else []

        with pf_tmp.open("a", encoding="utf-8") as fhout, pf_tmp_failed.open("a", encoding="utf-8") as fhout_failed:
            for split_name, split_dataset in orig_dataset.items():
                if self.max_samples is not None:
                    split_dataset = split_dataset.select(range(self.max_samples))

                done_subset_idxs, num_done, failed_subset_idxs, num_failed = self._get_done_failed_subset_idxs(
                    already_done_df, failed_df, split_name
                )

                print(
                    f"Skipping {len(done_subset_idxs)} already done examples in {split_name}"
                    f"\nSkipping {len(failed_subset_idxs)} failed examples in {split_name}"
                )

                messages = self._prepare_messages(split_dataset, done_subset_idxs)

                if not messages:
                    continue

                print(f"Number of messages to do: {len(messages)}")

                for answer_response in (
                    pbar := tqdm(
                        self.text_generator.batch_query_messages(messages, **kwargs),
                        total=len(messages),
                    )
                ):
                    pbar.set_description(f"{split_name} ({num_done:,} ✓ | {num_failed:,} ✗)")

                    result_row = {
                        "split": split_name,
                        "column": self.output_column,
                        "idx": answer_response.job_idx,
                    }

                    if answer_response.error is None and answer_response.text_response is not None:
                        convo = answer_response.text_response.strip()
                        # Returns a list of dictionaries where each dictionary has 'role' and 'content' keys
                        gen_messages = extract_conversation_from_string(
                            convo,
                            user_id=self.user_id,
                            assistant_id=self.assistant_id,
                        )
                        if not gen_messages:
                            result_row["error"] = "Empty conversation"
                            self._write_row_to_fh(fhout_failed, result_row)
                            num_failed += 1

                        result_row[self.output_column] = gen_messages
                        convos.append(result_row)
                        self._write_row_to_fh(fhout, result_row)
                        num_done += 1
                    else:
                        result_row["error"] = str(answer_response.error)
                        self._write_row_to_fh(fhout_failed, result_row)
                        num_failed += 1

        if convos:
            output_datasets = self._postprocess_dataset(convos, orig_dataset, self.output_column)
            return output_datasets
        else:
            return None

    def _prepare_messages(self, dataset, done_subset_idxs):
        """
        Prepares the messages to send to the API.
        :param dataset: dataset to prepare messages for
        :param done_subset_idxs: subset indices that have already been done
        :return:
        """
        persona_descriptions = list(self.personas.values())
        messages = [
            (
                sample_idx,
                (
                    [
                        {
                            "role": "system",
                            "content": self.system_prompt.format(
                                persona=choice(persona_descriptions), subject=sample[self.seed_column]
                            ),
                        }
                        if "{persona}" in self.system_prompt
                        else {"role": "system", "content": self.system_prompt.format(subject=sample[self.seed_column])}
                    ]
                ),
            )
            for sample_idx, sample in enumerate(dataset)
            if sample_idx not in done_subset_idxs
        ]

        return messages
