import re
import sys
import unicodedata as ud

import fasttext
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from InquirerPy import inquirer


latin_letters = {}  # memo


# Taken from https://stackoverflow.com/a/3308844/1150683
def is_latin(unicode_chr: str):
    try:
        return latin_letters[unicode_chr]
    except KeyError:
        return latin_letters.setdefault(unicode_chr, "LATIN" in ud.name(unicode_chr))


def only_roman_chars(unicode_text: str):
    return all(is_latin(uchr) for uchr in unicode_text if uchr.isalpha())


model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)


def identify_language(row, column_name: str):
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

    return label.replace("__label__", "")


def filter_dataset(row, column_names: list[str]):
    # Will be formatted like "als [een] {}", "ben [een] {}", "{} ben"
    ai_names = [
        "AI-assistent",
        "AI-gebaseerde assistent",
        "virtuele assistent",
        "digitale assistent",
        "tekst-assistent",
        "AI tekstgebaseerde asssistent",
        "tekstgebaseerde asssistent",
        "assistent",
        "taalmodel",
        "AI-taalmodel",
        "AI taalmodel",
    ]
    regex_names = re.compile(
        r"|".join([rf"als (?:een )?{name}|ben (?:een )?{name}|{name} ben" for name in ai_names]),
        flags=re.IGNORECASE,
    )

    brands = [
        "ChatGPT",
        "Chat GPT",
        "GPT3",
        "GPT 3",
        "gpt-3",
        "gpt-3.5-turbo",
        "GPT4",
        "GPT 4",
        "gpt-4",
        "gpt-4-turbo",
        "OpenAI",
        "ShareGPT",
    ]
    regex_brands = re.compile(r"|".join(brands), flags=re.IGNORECASE)

    knowledge_cut_offs = [
        "kennisafsluiting in 2023",
        "kennisstop in 2023",
        "kennisafsnijdatum van 2023",
        "cutoff in 2023",
        "Tot mijn kennis die bijgewerkt is tot begin 2023",
        "Voor zover mijn kennis reikt tot 2023",
        "Vanaf mijn kennis tot begin 2023",
        "As of my last update in 2023",
    ]
    regex_knowledge = re.compile(r"|".join(knowledge_cut_offs), flags=re.IGNORECASE)

    # "assistant" is an indicator of English output, the Dutch word is with an 'e': "assistent"
    incorrect_language = ["It seems like there was a typo", "assistant"]
    regex_language = re.compile(r"|".join(incorrect_language), flags=re.IGNORECASE)

    apologies = ["spijt me", "spijt mij", "sorry", "mijn excuses"]
    regex_apologies = re.compile(r"|".join(apologies), flags=re.IGNORECASE)

    for column_name in column_names:
        text = row[column_name]

        # When this is a conversation column with lists of dictionaries of the type {"content": ..., "role"}
        # we just glue everything together
        if isinstance(text, list):
            text = " ".join(msg["content"] for msg in text)

        text = text.replace("\n", " ")
        text = " ".join(text.split())

        if regex_names.search(text):
            return False

        if regex_brands.search(text):
            return False

        if regex_knowledge.search(text):
            return False

        if regex_language.search(text):
            return False

        if regex_apologies.search(text):
            return False

        # This is a very strict filter...
        if not only_roman_chars(text):
            return False

        if f"{column_name}_lid" not in row:
            lid = identify_language(row, column_name)
        else:
            lid = row[f"{column_name}_lid"]

        if lid != "nld_Latn":
            num_tokens = len(text.split())
            # If the language is not the required one, but the text is only short, it might just be "Ja." or "Nee."
            # or something that was tough to identify
            if num_tokens > 3:
                return False

    return True


def conv_to_content(sample, _col: str):
    conv = sample[_col]
    text = ""
    for turn in conv:
        if turn["role"] == "assistant":
            text += turn["content"] + " "

    return {f"{_col}_content": text.strip()}


def remove_duplicates(dataset, column_names: list[str]):
    original_size = dataset.shape[0]

    col_map = {}
    for col in column_names:
        if not isinstance(dataset[col][0], str):
            dataset = dataset.map(conv_to_content, fn_kwargs={"_col": col}, num_proc=64)
            col_map[col] = f"{col}_content"
        else:
            col_map[col] = col

    df = dataset.to_pandas()
    for col in column_names:
        df = df.drop_duplicates(subset=[col_map[col]]).reset_index(drop=True)
    df = df.drop(columns=[col_map[col] for col in column_names if col_map[col] != col])
    dataset = Dataset.from_pandas(df)

    dedupe_size = dataset.shape[0]
    print(f"Removed {original_size - dedupe_size} duplicates. New size: {dedupe_size}")

    return dataset


def main():
    api = HfApi()

    ds_name = inquirer.text(
        message="Dataset repo name:",
    ).execute()
    ds_config_name = inquirer.text(
        message="Dataset config name:",
        default="default",
    ).execute()
    ds_branch_name = inquirer.text(
        message="Dataset branch name:",
        default="main",
    ).execute()

    ds = load_dataset(ds_name, ds_config_name, revision=ds_branch_name)

    output_dataset = {}
    for split_name, split_ds in ds.items():
        print(
            f"Config: {ds_config_name}, Branch: {ds_branch_name}, split: {split_name}, dataset size: {split_ds.shape}"
        )

        dedupe_cols = inquirer.select(
            message="Select columns to deduplicate on:",
            choices=split_ds.column_names,
            multiselect=True,
            transformer=lambda result: f"{len(result)} column{'s' if len(result) > 1 else ''} selected ({', '.join(result)})",
        ).execute()
        split_ds = remove_duplicates(split_ds, dedupe_cols)

        lid_cols = inquirer.select(
            message="Select columns whose text to filter:",
            choices=split_ds.column_names,
            multiselect=True,
            transformer=lambda result: f"{len(result)} column{'s' if len(result) > 1 else ''} selected ({', '.join(result)})",
        ).execute()

        split_ds = split_ds.filter(lambda row: filter_dataset(row, lid_cols), keep_in_memory=True)
        output_dataset[split_name] = split_ds
        print("After filtering/deduplicating:", output_dataset[split_name].shape)

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

        config_name = inquirer.text(
            message=f"Config name to upload to in {repo_id}:",
            default="default",
        ).execute()

        revision_id = inquirer.text(message=f"Branch ID to upload to in {repo_id}:", default="main").execute()

        try:
            output_dataset.push_to_hub(repo_id, config_name, revision=revision_id)
        except Exception as exc:
            print(
                f"Could not upload the dataset ({str(exc)}). Make sure that the dataset and revsion are valid.",
                file=sys.stderr,
            )
        else:
            break


if __name__ == "__main__":
    main()
