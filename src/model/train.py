"""
Point d'entrée pour l'entraînement du modèle.

Charge les annonces récentes depuis la base, les passe par le nettoyage
(src/features/cleaning.py), encode les variables catégorielles, entraîne
un RandomForestRegressor, et sauvegarde le modèle + la structure de
colonnes attendue (nécessaire pour prédire sur de nouvelles données plus tard).

Usage :
    python -m src.model.train
"""

import sys
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.settings import DB_PATH
from src.database.repository import get_recent_annonces
from src.database.schema import init_db
from src.features.cleaning import build_clean_dataframe

MODEL_PATH = "models/model_v1.pkl"
COLUMNS_PATH = "models/model_v1_columns.pkl"
DAYS_WINDOW = 60       # fenêtre d'annonces "récentes" utilisées pour l'entraînement
TEST_SIZE = 0.2        # 20% des données réservées à l'évaluation
RANDOM_STATE = 0      # graine fixe pour des résultats reproductibles


def load_dataset(days: int = DAYS_WINDOW) -> pd.DataFrame:
    """Charge et nettoie les annonces des `days` derniers jours."""
    conn = init_db(DB_PATH)
    rows = get_recent_annonces(conn, days=days)
    conn.close()
    return build_clean_dataframe(rows)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare le DataFrame nettoyé en features (X) et cible (y),
    puis encode les colonnes catégorielles en colonnes binaires.
    """
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    y = df["price"]
    X = df.drop(columns=["price", "id"])

    # Transforme l'année d'immatriculation en âge du véhicule (plus robuste dans le temps)
    if "regdate" in X.columns:
        X["age"] = datetime.now().year - X["first_release_year"]
        X = X.drop(columns=["first_release_year"])

    # Colonnes numériques mais en réalité catégorielles (codes LeBonCoin sans
    # ordre réel), à forcer en catégoriel avant l'encodage
    FALSELY_NUM_COLS = ["gearbox", "fuel"]
    for col in FALSELY_NUM_COLS:
        if col in X.columns:
            X[col] = X[col].astype("str")

    # One-hot encoding : chaque colonne texte devient plusieurs colonnes 0/1.
    # Les colonnes déjà numériques (kilométrage, année...) passent inchangées.
    # dummy_na=True garde les valeurs manquantes comme leur propre catégorie
    # plutôt que de faire planter l'encodage.
    X = pd.get_dummies(X, dummy_na=True)

    return X, y


def train_model():
    df = load_dataset()
    print(f"{len(df)} annonces loaded for training.")

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    # On sauvegarde aussi la liste des colonnes : le one-hot encoding dépend
    # des catégories vues à l'entraînement, il faudra ré-aligner toute
    # nouvelle donnée sur cette même structure pour prédire correctement.
    joblib.dump(list(X.columns), COLUMNS_PATH)

    print(f"Model saved in {MODEL_PATH}")
    print(f"Training length: {len(X_train)} | Test length: {len(X_test)}")

    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_model()
