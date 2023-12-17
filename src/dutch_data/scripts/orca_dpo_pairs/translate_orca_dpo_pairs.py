from typing import Annotated, Optional

import typer
from dutch_data.azure_utils import Credentials
from dutch_data.azure_utils.querier import CyclicalAzureQuerier
from dutch_data.dataset_processing import SYSTEM_TRANSLATION_PROMPT
from dutch_data.dataset_processing.translate_hf_dataset import TranslateHFDataset
from dutch_data.text_generator import AzureTextGenerator, HFTextGenerator
from typer import Argument, Option


# NOTE: this prompt has around 1250 tokens so it will be quite expensive to run. It is relatively robust, however.
# This was written at the start of the project. In retro-spect it would have been much smarter to have separate prompts
# to translate system messages vs. user messages. This functionality exists (TranslateHFDataset.system_prompt accepts
# a dictionary of colname: sys_prompt to have separate ones), but for reproducibility we keep the default script like
# it is now.
_TRANSLATION_PROMPT = """Je bent een vertaler die Engels naar Nederlands vertaalt in hoge kwaliteit. Je vermijdt letterlijke vertalingen. Je zorgt voor een vloeiende en leesbare tekst. Je vermijdt bias, zoals gender bias en regionale bias. Je gebruikt informeel Standaardnederlands. Je doet ook aan lokalisatie, waar je je vertaling waar nodig aanpast aan de Nederlandse taal en de cultuur binnen het Nederlandstalige taalgebied (Nederland, Vlaanderen, en daarbuiten).

Hoofddoel: Vertaal alle gebruikersvragen naar het Nederlands, met uitzondering van specifieke gebruikersvragen die vertaalopdrachten bevatten.

Vertalen, Niet Beantwoorden: Bij elke gebruikersvraag, ongeacht de inhoud of vorm (zoals bevelen, vragen, raadsels, oefeningen, of vertaalopdrachten), vertaal je de vraag naar het Nederlands zonder deze inhoudelijk te beantwoorden of op te lossen.

Behoud van Broninformatie bij Vertaalopdrachten: Wanneer je een vertaalinstructie tegenkomt binnen een gebruikersvraag, vertaal je de instructie en behoud je de originele tekst in de brontaal. Vertaal de brontekst in de andere taal niet, maar kopieer deze naar je output in de oorspronkelijke taal.

<voorbeelden>

Gebruikersvraag: "Write a poem about spring in French."
Jouw reactie: Schrijf een gedicht over de lente in het Frans.
Voorbeeldinfo: generieke instructie met een specifieke taal.

Gebruikersvraag: "You are an AI assistant. Provide a detailed answer so user don’t need to search outside to understand the answer."
Jouw reactie: Je bent een AI-assistent. Geef een gedetailleerd antwoord zodat de gebruiker niet verder moet zoeken om het antwoord te begrijpen.
Voorbeeldinfo: vertaling van een generieke system-message voor een AI-assistent

Gebruikersvraag: "Summarize the following news article about space exploration. [news article]"
Jouw reactie: Vat het volgende nieuwsartikel over ruimteverkenning samen. [nieuwsartikel]
Voorbeeldinfo: generieke instructie zonder een specifieke taal. Merk op dat de tekst simpelweg vertaald is, en dat de instructie om tekst samen te vatten niet werd gevolgd maar enkel vertaald. Het nieuwsartikel werd ook helemaal overgenomen in de doeltekst.

Gebruikersvraag: "В процессе разучивания и шлифования программ Translate to English."
Jouw reactie: В процессе разучивания и шлифования программ Vertaal naar het Engels.
Voorbeeldinfo: een niet-Engelse taal werd gegeven, gevolgd door de instructie om te vertalen naar het Engels. De niet-Engelse taal moet behouden blijven.

Gebruikersvraag: "What is the correct way to use 'their', 'they're', and 'there'?"
Jouw reactie: Wat is de correcte manier om 'their', 'they're' en 'there' in het Engels te gebruiken?
Voorbeeldinfo: in de gebruikersvraag staat een grammaticale vraag. In de vertaling wordt ook de vraag over het Engels overgenomen, maar wordt expliciet toegevoegd dat het om Engelse grammatica gaat.

Gebruikersvraag: "Can you tell me how to say 'pineapple' in French and German?"
Jouw reactie: Kun je me vertellen hoe je 'ananas' zegt in het Frans en Duits?
Voorbeeldinfo: In de gebruikersvraag staat "pineapple" in het Engels, wat naar het Nederlands vertaald werd omdat er geen specifieke brontaal vermeld werd.

Gebruikersvraag: "Fix this sentence's grammar: 'He don't know the answer'"
Jouw reactie: Verbeter deze zin: 'Hij weet niet de antwoord'
Voorbeeldinfo: Er werd een algemene instructie gegeven met een taalfout in het Engels. Dit werd niet letterlijk vertaald maar aangepast naar het Nederlands, met een gelijkaardige grammaticale fout in het Nederlands.

Gebruikersvraag: "Translate this text to German: We never knew his name. But I knew his father back in high school. German:'"
Jouw reactie: Vertaal deze tekst naar het Duits: We wisten zijn naam niet maar ik kende zijn vader vanop school. Duits:
Voorbeeldinfo: De brontaal werd niet gespecificeerd, enkel de doeltaal (Duits). Daarom werd de brontekst naar het Nederlands vertaald.

Gebruikersvraag: "Translate this English text to German: We never knew his name. But I knew his father back in high school. German:'"
Jouw reactie: Vertaal deze Engelse tekst naar het Duits: We never knew his name. But I knew his father back in high school. Duits:
Voorbeeldinfo: De brontaal werd gespecificeerd (English). Daarom werd de brontekst in het Engels behouden. Enkel de instructie werd vertaald.

</voorbeelden>

Belangrijk: Blijf altijd trouw aan de oorspronkelijke intentie van de gebruikersvraag tijdens het vertalen en zorg ervoor dat de brontekst behouden blijft bij specifieke vertaalopdrachten."""


app = typer.Typer()


@app.command()
def translate_orcadpo_system_question(
    output_directory: Annotated[str, Argument(help="output directory to save the translated dataset to")],
    hf_model_name: Annotated[
        Optional[str],
        Option(
            help="HuggingFace model name to use for the text generator. Note that currently only conversational style models are supported"
        ),
    ] = None,
    credentials_file: Annotated[Optional[str], Option(help="JSON file containing credentials")] = None,
    credentials_profiles: Annotated[
        Optional[list[str]], Option(help="which credential profile(s) (key) to use from the credentials file")
    ] = None,
    tgt_lang: Annotated[
        str,
        Option(
            help="target language to translate to. If 'Dutch', will use a Dutch system message, otherwise an English one"
        ),
    ] = "Dutch",
    *,
    output_hub_name: Annotated[
        Optional[str],
        Option(
            help="optional hub name to push the translated dataset to. Should start with an org or username,"
            " e.g. 'MyUserName/my-dataset-name'"
        ),
    ] = None,
    output_hub_revision: Annotated[
        Optional[str],
        Option(help="hub branch to upload to. If not specified, will use the default branch, typically 'main'."),
    ] = None,
    max_workers: Annotated[
        int, Option("--max-workers", "-j", help="(azure) how many parallel workers to use to query the API")
    ] = 6,
    max_retries: Annotated[int, Option(help="(azure) how many times to retry on errors")] = 3,
    max_tokens: Annotated[int, Option(help="max new tokens to generate")] = 2048,
    timeout: Annotated[float, Option("--timeout", "-t", help="(azure) timeout in seconds for each API call")] = 30.0,
    verbose: Annotated[
        bool, Option("--verbose", "-v", help="(azure) whether to print more information of the API responses")
    ] = False,
):
    """
    Translate the 'system' and 'question' columns of the Intel/orca_dpo_pairs dataset to a given language
    (default Dutch). Depending on the given arguments, will use either a HuggingFace model or the Azure API to
    translate the dataset. Will save the translated dataset to the given output directory. Optionally, will also
    upload the translated dataset to the given hub name and revision.
    """
    sys_msg = _TRANSLATION_PROMPT if tgt_lang.lower() == "dutch" else SYSTEM_TRANSLATION_PROMPT

    if hf_model_name is None and credentials_file is None:
        raise ValueError("Either hf_model_name or credentials_file must be given")

    if hf_model_name:
        text_generator = HFTextGenerator(hf_model_name)
    else:
        if len(credentials_profiles) == 1:
            credentials = Credentials.from_json(credentials_file, credentials_profiles[0])
            text_generator = AzureTextGenerator.from_credentials(
                credentials, max_workers=max_workers, timeout=timeout, max_retries=max_retries, verbose=verbose
            )
        else:
            cyclical_querier = CyclicalAzureQuerier.from_json(
                credentials_file,
                credentials_profiles,
                max_workers=max_workers,
                timeout=timeout,
                max_retries=max_retries,
                verbose=verbose,
            )
            text_generator = AzureTextGenerator(cyclical_querier)

    translator = TranslateHFDataset(
        text_generator=text_generator,
        dataset_name="Intel/orca_dpo_pairs",
        tgt_lang=tgt_lang,
        dout=output_directory,
        columns=["system", "question"],
        output_hub_name=output_hub_name,
        output_hub_revision=output_hub_revision,
        merge_with_original=True,
        system_prompt=sys_msg,
        verbose=verbose,
    )

    if hf_model_name:
        return translator.process_dataset(max_new_tokens=max_tokens)
    else:
        return translator.process_dataset(max_tokens=max_tokens)
