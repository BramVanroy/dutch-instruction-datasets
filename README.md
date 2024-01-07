# Dutch instruction dataset creation

In this repository scripts are provided to build your own instruction dataset through OpenAI services. We specifically
make use of Azure services.

NOTE: this is a work in progress. The scripts are prone to change so do not use this in production. They are also not
as heavily optimized as I would like them to be: they do not support batching, for example. 

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
you. It will also save intermediate results and can simply restart where it left off.


Example usage:

```shell
answer-hf --help
```

## TODO

- [ ] Add separate test for Azure Generator:
  - [ ] Test different scenarios (exception, stop, content filter...)
- [ ] Add support for OpenAI API (not Azure)
- [ ] Add tests for conversation
- [ ] Add tests for translation
- [ ] Add filtering/post-processing


## License

Licensed under [GPLv3](LICENSE). 
