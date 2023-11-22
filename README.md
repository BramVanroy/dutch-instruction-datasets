# Dutch instruction dataset creation

In this repository scripts are provided to build your own instruction dataset through OpenAI services. We specifically
make use of Azure services.

The creation is inspired by [UltraChat](https://github.com/thunlp/UltraChat), a dataset for dialogue generation.

## Usage

### `generate-subtopics`

In a first step, we generate subtopics for a given topic. Based on your credentials and Azure config, the API is 
queried in parallel to suggest a list of subtopics. Those will be saved on a JSON file when the `--output-file`
parameter is provided, otherwise the output is printed to the console.

Example usage:

```shell
generate-subtopics .credentials.json gpt-35-turbo --num-topics 5 --output-file data/world-subtopics.json
```

## License

Licensed under [GPLv3](LICENSE). 
