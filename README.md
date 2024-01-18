# Dutch instruction dataset creation

In this repository scripts are provided to build your own instruction dataset through OpenAI services. We specifically
make use of Azure services.


## Usage

If you use the Azure services in the following scripts, you will need to specify a credentials file. This file should
have the following structure, where each key is a "profile", like "gpt-4". In the examples below, this has been
saved to a file called `.credentials.json`.

```json
{
    "gpt-4": {
        "endpoint": "https://abc.openai.azure.com/",
        "api_key": "[secret-key]",
        "api_version": "2023-07-01-preview",
        "deployment_name": "deployment-name1"
    },
    "gpt-35-turbo": {
        "endpoint": "https://def.openai.azure.com/",
        "api_key": "[secret-key]",
        "api_version": "2023-07-01-preview",
        "deployment_name": "deployment-name2"
    }
}
```

For all commands a `--help` option is available with more explanations about all the arguments.

### `interactive-query`

Launch an interactive query session. This will allow you to query the OpenAI API and "talk" to the model. This 
implementation is not very smart, and will not do any smart length filtering when you exceed the context window.
So do not use it for extended conversations.

It supports both Azure services and Hugging Face models.

Example usage Azure with the `gpt-35-turbo` profile:

```shell
interactive-query azure .credentials.json gpt-35-turbo
```

Example usage Hugging Face with the `BramVanroy/Llama-2-13b-chat-dutch` model (`transformers` must be installed,
and for many options w.r.t. quantization you will also need `accelerate` and `bitsandbytes`):

```shell
interactive-query huggingface BramVanroy/Llama-2-13b-chat-dutch --load-in-8bit
```


### `translate-hf`

Most of the time we want to start with translating system message and/or user messages, and then "answer" those later
on in a next step. `translate-hf` is the entry point to translate specific columns and splits of any dataset on the
Hugging Face hub. It will save the translated dataset to a temporary location, and then upload it to the hub.

It should be relatively robust as it saves intermediate results and can simply restart where it left off.

Example usage:

```shell
translate-hf HuggingFaceH4/ultrachat_200k data/ultrachat_200k/ultrachat_200k-gpt-4-turbo-translated --split train_sft --split test_sft --columns prompt --src-lang English --tgt-lang Dutch --output-hub-name BramVanroy/ultrachat_200k_dutch --output-hub-revision 1-gpt-4-turbo-translated -j 8 --system-prompt .transl_sysprompt_en-nl
```

This will:

- Translate the `train_sft` and `test_sft` splits of `HuggingFaceH4/ultrachat_200k` from English to Dutch
- It will save temporary results to `data/ultrachat_200k/ultrachat_200k-gpt-4-turbo-translated`
- It will upload the final dataset to revision (branch) `1-gpt-4-turbo-translated` in the `BramVanroy/ultrachat_200k_dutch` dataset
- It will use 8 processes to speed up the translation
- It will use the `.transl_sysprompt_en-nl` file that contains a system prompt as the system message

### `answer-hf`

In the next step we want to use models or APIs to generate an answer to given columns. This script will do that for
you. The only required input that is used is the given `user-column` as the user message, optionally a `system-column`,
and the model answer to those will be saved into the `response-column` (defautls to `response`).



Example usage:

```shell
answer-hf --help
```

### `conversation-hf`

This script allows you to build a conversation in a single model response. Importantly, the specified system_prompt is
supposed to tell the model to create a multi-turn conversation and also give an example of such a conversation, with
specified identifiers for the user and assistant in the generated conversation. These identifiers should also be given in
this script (defaults to `user: ` and `assistant: `).

You can also specifiy personas with `--personas` which should be a JSON file containinga main key `personas` with
persona names and their descriptions, which can then be passed to the system_prompt as long as it has a `{persona}`
field in its text. The JSON file can optionally also have a `weights` key, which indicates how randomly weighted
the different personas are chosen. If not given, all personas are equally likely. To repeat: when you provide a 
`personas` file, the persona descriptions will be randomly selected for each sample (optionally weighted) and
plugged into the system_prompt that you provided as long as that text (file) contains the string `{persona}`.

Example usage:

```shell
answer-hf --help
```

### `interactive-lid`

An interactive script to add language identification to specified columns in your dataset.
The script handles `messages` (lists of dictionaries) by simply concatenating all content keys.

The script will add `{colname}_lid` and `{colname}_lid_prob` columns to your dataset.

Usage: simply run `interactive-lid` and follow instructions.


### `interactive-filter-dutch`

An interactive script to filter out non-Dutch messages from your dataset. It does so based on the columns
added with `interactive-lid` so that script should be used first.

In addition to language filtering, it also allows you to filter out messages with specific characteristics. Text matching occurs in a case-insensitive manner.

- messages with non-Roman characters are removed (every character must have "LATIN" in its unicode name; note that this solution is not flawless: https://stackoverflow.com/a/3308844/1150683)
This is a very strict filter and will lead to the removal of data that you may have wanted to keep (e.g. messages that involve a translation task to non-Latin script languages)
- messages that are not identified as Dutch and that are longer than three white-space separated tokens are removed
- any text containing "spijt me", "spijt mij", "sorry", "mijn excuses", because those often indicate that the system could not successfully reply to a request
- any text containing "It seems like there was a typo", "assistant", because those often indicate that the system could not successfully reply to a request. Note that `assistant` is the English word (`assistent` is Dutch), so when `assistant` appears something is likely wrong
- any text indicating knowledge cut-offs:
    - kennisafsluiting in 2023 
    - kennisstop in 2023 
    - kennisafsnijdatum van 2023 
    - cutoff in 2023 
    - Tot mijn kennis die bijgewerkt is tot begin 2023 
    - Voor zover mijn kennis reikt tot 2023 
    - Vanaf mijn kennis tot begin 2023 
    - As of my last update in 2023
- any text referencing other language models
    - ChatGPT
    - Chat GPT
    - GPT3
    - GPT 3
    - gpt-3
    - gpt-3.5-turbo
    - GPT4
    - GPT 4
    - gpt-4
    - gpt-4-turbo
    - OpenAI
    - ShareGPT
- any self-referencing text about being a language model. This often indicates that a model is not capable of a specific task, in case we drop those samples to instead focus on the tasks that it can do.
The following strings are matched in a template for all occurrences of "als [een] {}", "ben [een] {}", "{} ben"
    - AI-assistent
    - AI-gebaseerde assistent
    - virtuele assistent
    - digitale assistent
    - tekst-assistent
    - AI tekstgebaseerde asssistent
    - tekstgebaseerde asssistent
    - assistent
    - taalmodel
    - AI-taalmodel
    - AI taalmodel
    - 
## License

Licensed under [GPLv3](LICENSE). 
