"""
Load the data from the CSV and return it in an application-friendly format.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict
import csv
import json

CSV_FILE_PATH = "index.csv"
METADATA_FILE_PATH = "metadata.csv"
JSON_FILE_PATH = "index.json"


@dataclass
class Drink:
    name: str
    page: str
    ingredients: List[str]


@dataclass
class Ingredient:
    name: str
    bottle: bool  # is this a bottle or an ingredient
    recipe: bool  # is this a full recipe (i.e. make this syrup, infuse x, etc)
    price: str  # price I looked up


def get_drinks() -> List[Drink]:
    """
    Load all drinks from the CSV.
    """

    drinks: Dict[str, Drink] = {}
    ingredients: Dict[str, Ingredient] = {}

    with open(METADATA_FILE_PATH, mode="r", encoding="utf-8") as metadata_file:
        for row in csv.DictReader(metadata_file):
            name = row["ingredient"]
            bottle = row["bottle"] == "TRUE"
            recipe = row["recipe"] == "TRUE"

            if name not in ingredients:
                ingredients[name] = Ingredient(
                    name=name, price=row["price"], bottle=bottle, recipe=recipe
                )

    with open(CSV_FILE_PATH, mode="r", encoding="utf-8") as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            name = row["Drink Name"]

            if name not in drinks:
                drinks[name] = Drink(name=name, page=row["Page"], ingredients=[])

            drinks[name].page = row["Page"]

            # filter to *just* bottles for now:
            ingredient = row["Ingredient"]
            if ingredients[ingredient].bottle:
                drinks[name].ingredients.append(ingredient)

    # Saving the list of objects to a JSON file, just for easy inspecting.
    with open(JSON_FILE_PATH, mode="w", encoding="utf-8") as jsonfile:
        json.dump([asdict(x) for x in drinks.values()], jsonfile, indent=2)

    return list(drinks.values())
