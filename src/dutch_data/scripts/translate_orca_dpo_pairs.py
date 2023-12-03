from typing import Annotated, Optional

import typer
from dutch_data.translate_hf import SYSTEM_TRANSLATION_PROMPT, translate_hf_dataset
from typer import Argument, Option


# NOTE: this prompt has around 1250 tokens so it will be quite expensive to run. It is relatively robust, however.
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
Voorbeeldinfo: In de gebruikersvraag staat "apple" in het Engels, wat naar het Nederlands vertaald werd omdat er geen specifieke brontaal vermeld werd.

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
    credentials_file: Annotated[str, typer.Argument(help="JSON file containing credentials")],
    credentials_profile: Annotated[
        str, Argument(help="which credential profile (key) to use from the credentials file")
    ],
    output_directory: Annotated[str, Argument(help="output directory to save the translated dataset to")],
    tgt_lang: Annotated[
        str,
        Option(
            help="target language to translate to. If 'Dutch', will use a Dutch system message, otherwise an English one"
        ),
    ] = "Dutch",
    *,
    hub_name: Annotated[
        Optional[str],
        Option(
            help="optional hub name to push the translated dataset to. Should start with an org or username,"
            " e.g. 'MyUserName/my-dataset-name'"
        ),
    ] = None,
    max_num_workers: Annotated[int, Option(help="how many parallel workers to use to query the API")] = 6,
    max_tokens: Annotated[int, Option(help="max new tokens to generate with the API")] = 2048,
):
    sys_msg = _TRANSLATION_PROMPT if tgt_lang.lower() == "dutch" else SYSTEM_TRANSLATION_PROMPT

    return translate_hf_dataset(
        dataset_name="Intel/orca_dpo_pairs",
        tgt_lang=tgt_lang,
        credentials_file=credentials_file,
        credentials_profile=credentials_profile,
        dout=output_directory,
        columns=["system", "question"],
        hub_name=hub_name,
        merge_with_original=True,
        max_num_workers=max_num_workers,
        system_prompt_template=sys_msg,
        max_tokens=max_tokens,
        timeout=360.0,
    )
