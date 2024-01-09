import sys

import fasttext
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from InquirerPy import inquirer


def identify_language(row, column_name: str, model):
    text = row[column_name]

    # When this is a conversation column with lists of dictionaries of the type {"content": ..., "role"}
    # we just glue everything together
    if isinstance(text, list):
        text = " ".join(msg["content"] for msg in text)

    text = text.replace("\n", " ")
    text = " ".join(text.split())
    if text is None:
        return None

    label, prob = model.predict(text, k=1)
    label = label[0]

    return {f"{column_name}_lid": label.replace("__label__", ""), f"{column_name}_lid_prob": prob.item()}


def main():
    api = HfApi()

    while True:
        ds_name = inquirer.text(
            message="Dataset repo name:",
        ).execute()

        try:
            refs = api.list_repo_refs(ds_name, repo_type="dataset")
        except RepositoryNotFoundError:
            print(
                "Dataset not found. Try again and make sure that you are logged in with `huggingface-cli login`"
                " when accessing a private dataset.",
                file=sys.stderr,
            )
        else:
            break

    ds_branches = [ref.name for ref in refs.branches]

    branch_name = inquirer.select(
        message="Select branch:",
        choices=ds_branches,
    ).execute()

    ds = load_dataset(ds_name, revision=branch_name)

    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    model = fasttext.load_model(model_path)

    output_dataset = {}
    for split_name, split_ds in ds.items():
        print(f"Branch: {branch_name}, split: {split_name}, dataset size: {split_ds.shape}")

        lid_cols = inquirer.select(
            message="Select columns whose text to language identify:",
            choices=split_ds.column_names,
            multiselect=True,
            transformer=lambda result: f"{len(result)} column{'s' if len(result) > 1 else ''} selected ({', '.join(result)})",
        ).execute()

        for col in lid_cols:
            split_ds = split_ds.map(lambda row: identify_language(row, col, model), keep_in_memory=True)
        output_dataset[split_name] = split_ds
    output_dataset = DatasetDict(output_dataset)

    while True:
        output_path = inquirer.text(
            message="Path to save the merged dataset to (Enter to skip):", default=""
        ).execute()

        if not output_path:
            break

        try:
            output_dataset.save_to_disk(output_path)
        except OSError as exc:
            print(f"Could not save files on disk ({str(exc)}). Specify a valid path.", file=sys.stderr)
        else:
            break

    while True:
        repo_id = inquirer.text(message="Hugging Face Hub repo ID to upload to (Enter to skip):", default="").execute()

        if not repo_id:
            break

        revision_id = inquirer.text(message=f"Branch ID to upload to in {repo_id}:", default="main").execute()

        try:
            output_dataset.push_to_hub(repo_id, revision=revision_id)
        except Exception as exc:
            print(
                f"Could not upload the dataset ({str(exc)}). Make sure that the dataset and revsion are valid.",
                file=sys.stderr,
            )
        else:
            break


if __name__ == "__main__":
    main()
