import sys
from functools import reduce

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from InquirerPy import inquirer


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

    selected_branches = inquirer.select(
        message="Select branches:",
        choices=ds_branches,
        multiselect=True,
        transformer=lambda result: f"{len(result)} branch{'es' if len(result) > 1 else ''} selected ({', '.join(result)})",
    ).execute()

    ds_revisions_d = {branch_name: load_dataset(ds_name, revision=branch_name) for branch_name in selected_branches}

    for branch_name, ds in ds_revisions_d.items():
        for split_name, split_ds in ds.items():
            print(f"Branch: {branch_name}, split: {split_name}, dataset size: {split_ds.shape}")

    ds_revisions = list(ds_revisions_d.values())

    if not all(set(ds_revisions[0]) == set(r) for r in ds_revisions):
        print("Branches do not have the same splits (like train, dev, test). Only keeping the common splits.")

    intersected_splits = list(reduce(set.intersection, [set(r.column_names) for r in ds_revisions]))

    if not intersected_splits:
        print("No common splits found. Exiting.", file=sys.stderr)
        exit(1)

    output_dataset = {}
    for split in intersected_splits:
        dfs_splits = [r[split].to_pandas() for r in ds_revisions]
        print(f"Processing split: {split}")
        intersected_cols = list(reduce(set.intersection, [set(df.columns) for df in dfs_splits]))
        merge_on_cols = inquirer.select(
            message="Select columns to merge on:",
            choices=intersected_cols,
            multiselect=True,
            transformer=lambda result: f"{len(result)} column{'s' if len(result) > 1 else ''} selected ({', '.join(result)})",
        ).execute()

        lens_before = [len(df.index) for df in dfs_splits]
        dfs_splits = [df.drop_duplicates(subset=merge_on_cols, keep="first") for df in dfs_splits]
        lens_after = [len(df.index) for df in dfs_splits]

        if lens_before != lens_after:
            print(
                f"Removed duplicate entries on the merge_on columns from all datasets."
                f" Sizes before: {lens_before}, sizes after: {lens_after}"
            )

        merged_df = reduce(lambda df1, df2: df1.merge(df2, on=merge_on_cols), dfs_splits)
        required_cols = [c for c in merged_df.columns if c not in merge_on_cols]
        merged_df = merged_df.dropna(subset=required_cols)
        print(f"Merged dataset size: {merged_df.shape}")
        print(merged_df.head())
        output_dataset[split] = Dataset.from_pandas(merged_df)

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
