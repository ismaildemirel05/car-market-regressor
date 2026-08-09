"""
Point d'entrée pour l'entraînement du modèle.

Charge les annonces récentes depuis la base, les passe par le nettoyage
(src/features/cleaning.py), encode les variables catégorielles
(src/features/encoding.py), entraîne un RandomForestRegressor, et
sauvegarde le modèle + les encoders (nécessaires pour prédire sur de
nouvelles données plus tard).

Usage :
    python -m src.model.train
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.settings import DB_PATH
from src.database.repository import get_recent_annonces
from src.database.schema import init_db
from src.features.cleaning import build_clean_dataframe
from src.features.encoding import apply_encoders, fit_encoders

MODEL_PATH = "models/model_v1.pkl"
ENCODERS_PATH = "models/model_v1_encoders.pkl"
DAYS_WINDOW = 60       # fenêtre d'annonces "récentes" utilisées pour l'entraînement
TEST_SIZE = 0.2        # 20% des données réservées à l'évaluation
RANDOM_STATE = 0       # graine fixe pour des résultats reproductibles

# Colonnes numériques mais en réalité catégorielles (codes LeBonCoin sans
# ordre réel), à forcer en catégoriel avant l'encodage
FALSELY_NUM_COLS = ["gearbox", "fuel"]


def load_dataset(days: int = DAYS_WINDOW) -> pd.DataFrame:
    """Charge et nettoie les annonces des `days` derniers jours."""
    conn = init_db(DB_PATH)
    rows = get_recent_annonces(conn, days=days)
    conn.close()
    return build_clean_dataframe(rows)


def train_model():
    df = load_dataset()
    print(f"{len(df)} annonces loaded for training.")

    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    for col in FALSELY_NUM_COLS:
        if col in df.columns:
            df[col] = df[col].astype("str")

    # Split BEFORE encoding : the encoders must only ever see the train
    # split, otherwise the target encoding leaks information from the
    # test set (and, later, from real-world data you're trying to predict).
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    encoders = fit_encoders(df_train, target_column="price")
    df_train = apply_encoders(df_train, encoders)
    df_test = apply_encoders(df_test, encoders)

    X_train = df_train.drop(columns=["price", "id"])
    y_train = df_train["price"]
    X_test = df_test.drop(columns=["price", "id"])
    y_test = df_test["price"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    # On sauvegarde les encoders : ils contiennent les catégories one-hot
    # et les moyennes de target encoding apprises sur le train. Il faudra
    # les réutiliser tels quels (apply_encoders) pour prédire correctement
    # sur toute nouvelle annonce.
    joblib.dump(encoders, ENCODERS_PATH)

    print(f"Model saved in {MODEL_PATH}")
    print(f"Training length: {len(X_train)} | Test length: {len(X_test)}")

    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_model()