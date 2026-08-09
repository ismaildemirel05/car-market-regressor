"""
Nettoyage / transformation des annonces brutes en données exploitables
pour l'entraînement du modèle.

Ce module lit les lignes stockées en base (dont attributs_json, qui contient
les attributs bruts renvoyés par LeBonCoin) et produit un DataFrame propre
et typé.

À COMPLÉTER : les clés dans ATTRIBUTE_KEYS ci-dessous doivent correspondre
aux vraies clés trouvées dans ton attributs_json (kilométrage, année,
carburant, boîte, puissance...). Inspecte un exemple réel en base pour
les identifier, puis remplis le mapping.
"""

import json
import re
from datetime import datetime

import pandas as pd

# --- Mapping "nom de colonne final" -> "clé réelle dans attributs_json" ---

FEATURES = {
    "mileage": ("mileage", float),
    "horse_power": ("horse_power_din", float),
    "gearbox": ("gearbox", str),
    "fuel": ("fuel", str),
    "car_brand": ("u_car_brand", str),
    "car_model": ("u_car_model", str),
    "car_version": ("u_car_version", str),
    "color": ("vehicule_color", str),
    "first_release_year": ("regdate", int),
    "doors": ("doors", int),
    "seats": ("seats", int),
    "vehicle_type": ("vehicle_type", str),
    "vehicle_emissions": ("vehicle_euro_emissions_standard", str),
    "vehicle_damage" : ("vehicle_damage", str),
}


def extract_attribute(attributs: dict, key: str):
    """Récupère une valeur dans le dict d'attributs, ou None si absente."""
    return attributs.get(key)


def to_numeric(value) -> int | float | None:
    """
    Convertit une valeur potentiellement sale (string avec espaces, unités...)
    en nombre. Retourne None si la conversion échoue.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value

    # Garde uniquement les chiffres (et le point décimal) dans la string
    nettoye = re.sub(r"[^\d.]", "", str(value))
    return float(nettoye) if nettoye else None


def clean_val(val: str | int | float, var_type: type) -> int | float | str | None:
    return to_numeric(val) if (var_type == int) or (var_type == float) else val


def clean_row(row: dict) -> dict:
    """
    Transforme une ligne brute de la base (dict) en ligne propre pour le modèle.
    `row` correspond à une ligne retournée par repository.get_recent_annonces().
    """
    attributes = json.loads(row["attributes_json"]) if row.get("attributes_json") else {}
    extracted = {feature: clean_val(attributes.get(key, None), var_type) for feature, (key, var_type) in FEATURES.items()}

    return {
        "id": row["id"],
        "price": to_numeric(row["price"]),
        # "region": row["region"],
        "department_num": int(to_numeric(row["zipcode"])//1000),
        **extracted
    }


def build_clean_dataframe(rows: list[dict]) -> pd.DataFrame:
    """
    Point d'entrée principal : prend les lignes brutes de la base
    et retourne un DataFrame nettoyé, prêt pour l'entraînement.
    """
    lignes_propres = [clean_row(row) for row in rows]
    df = pd.DataFrame(lignes_propres)

    # Filtres de base à ajuster selon tes besoins :
    df = df[df["price"] > 0]          # écarte les prix nuls/négatifs
    df = df.drop_duplicates(subset="id")

    # useless columns (unique values)
    df = df.drop(columns=["car_version"])

    # Transforme l'année d'immatriculation en âge du véhicule (plus robuste dans le temps)
    if "first_release_year" in df.columns:
        df["age"] = datetime.now().year - df["first_release_year"]
        df = df.drop(columns=["first_release_year"])

    return df


if __name__ == "__main__":

    from src.database.repository import get_recent_annonces
    from src.database.schema import init_db

    db_path = "data/annonces_voitures.db"
    conn = init_db(db_path)

    annonces = get_recent_annonces(conn, days=30)
    df = build_clean_dataframe(annonces)
    print(df["color"].unique())