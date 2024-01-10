import re
import sys
import unicodedata as ud

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
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

        if row[f"{column_name}_lid"] != "nld_Latn":
            num_tokens = len(text.split())
            # If the language is not the required one, but the text is only short, it might just be "Ja." or "Nee."
            # or something that was tough to identify
            if num_tokens > 3:
                return False

    return True


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

    output_dataset = {}
    for split_name, split_ds in ds.items():
        print(f"Branch: {branch_name}, split: {split_name}, dataset size: {split_ds.shape}")

        lid_cols = inquirer.select(
            message="Select columns whose text to language identify:",
            choices=split_ds.column_names,
            multiselect=True,
            transformer=lambda result: f"{len(result)} column{'s' if len(result) > 1 else ''} selected ({', '.join(result)})",
        ).execute()

        output_dataset[split_name] = split_ds.filter(lambda row: filter_dataset(row, lid_cols), keep_in_memory=True)
        print("After filtering:", output_dataset[split_name].shape)

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
