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


### `translate-orcadpo`

As a starting point, we want to translate the system messages and the questions of Intel's DPO dataset, which
they selected from OpenOrca. This script will translate the messages and questions from English to Dutch by default
but you can specify a different target language if you want to. You see here that `--hub-name` is used to specify
the dataset on the Hugging Face hub where we will upload the translated dataset to. Adapt this to your own profile.

Example usage:

```shell
translate-orcadpo .credentials.json gpt-35-turbo data/translated-orca-dpo --hub-name BramVanroy/orca_dpo_pairs_dutch
```


### `answer-orcadpo`

Finally, using the translations of the system message and the questions, we can query the model to get the answers
to the questions and save them to a file as a Hugging Face dataset. Note that the starting point is, again, a dataset
on the Hugging Face hub. This dataset is the translated version of the DPO dataset created in the previous step.

Example usage:

```shell
answer-orcadpo .credentials.json gpt-35-turbo BramVanroy/orca_dpo_pairs_dutch question_dutch data/answered-orca-dpo --system-column system_dutch
```


## License

Licensed under [GPLv3](LICENSE). 
