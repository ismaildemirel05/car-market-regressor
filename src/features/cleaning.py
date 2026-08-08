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

import pandas as pd

# --- Mapping "nom de colonne final" -> "clé réelle dans attributs_json" ---
ATTRIBUTE_KEYS = {
    "annee": "regdate",
    "kilometrage": "mileage",
    "carburant": "fuel",
    "boite": "gearbox",
    "puissance": "horse_power_din",
}


def extract_attribute(attributs: dict, key: str):
    """Récupère une valeur dans le dict d'attributs, ou None si absente."""
    return attributs.get(key)


def to_numeric(value) -> float | None:
    """
    Convertit une valeur potentiellement sale (string avec espaces, unités...)
    en nombre. Retourne None si la conversion échoue.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    # Garde uniquement les chiffres (et le point décimal) dans la string
    nettoye = re.sub(r"[^\d.]", "", str(value))
    return float(nettoye) if nettoye else None


def clean_row(row: dict) -> dict:
    """
    Transforme une ligne brute de la base (dict) en ligne propre pour le modèle.
    `row` correspond à une ligne retournée par repository.get_recent_annonces().
    """
    attributs = json.loads(row["attributs_json"]) if row.get("attributs_json") else {}

    return {
        "id": row["id"],
        "prix": to_numeric(row["prix"]),
        "marque": row["marque"],
        "region": row["region"],

        # TODO : une fois ATTRIBUTE_KEYS rempli, décommenter/adapter ces lignes
        # "annee": to_numeric(extract_attribute(attributs, ATTRIBUTE_KEYS["annee"])),
        # "kilometrage": to_numeric(extract_attribute(attributs, ATTRIBUTE_KEYS["kilometrage"])),
        # "carburant": extract_attribute(attributs, ATTRIBUTE_KEYS["carburant"]),
        # "boite": extract_attribute(attributs, ATTRIBUTE_KEYS["boite"]),
        # "puissance": to_numeric(extract_attribute(attributs, ATTRIBUTE_KEYS["puissance"])),
    }


def build_clean_dataframe(rows: list[dict]) -> pd.DataFrame:
    """
    Point d'entrée principal : prend les lignes brutes de la base
    et retourne un DataFrame nettoyé, prêt pour l'entraînement.
    """
    lignes_propres = [clean_row(row) for row in rows]
    df = pd.DataFrame(lignes_propres)

    # Filtres de base à ajuster selon tes besoins :
    df = df[df["prix"] > 0]          # écarte les prix nuls/négatifs
    df = df.drop_duplicates(subset="id")

    return df


if __name__ == "__main__":

    from src.database.repository import get_recent_annonces
    from src.database.schema import init_db

    db_path = "data/annonces_voitures.db"
    conn = init_db(db_path)

    annonces = get_recent_annonces(conn, days=30)
    df = build_clean_dataframe(annonces)
    print(df.head(5))