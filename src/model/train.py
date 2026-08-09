"""
Point d'entrée pour l'entraînement du modèle.

Charge les annonces récentes depuis la base, les passe par le nettoyage
(src/features/cleaning.py), puis fitte un Pipeline scikit-learn complet
(imputation -> target encoding -> one-hot encoding -> RandomForestRegressor)
défini dans src/features/preprocessing.py.

Le Pipeline entier (preprocessing + modèle) est sauvegardé en un seul
fichier : il suffira de le recharger avec joblib.load() pour prédire sur
de nouvelles annonces, sans avoir à rejouer manuellement le nettoyage
catégoriel ou l'imputation.

Usage :
    python -m src.model.train
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.settings import DB_PATH
from src.database.repository import get_recent_annonces
from src.database.schema import init_db
from src.features.cleaning import build_clean_dataframe
from src.features.preprocessing import build_preprocessing_pipeline

MODEL_PATH = "models/model_v1.pkl"
DAYS_WINDOW = 60       # fenêtre d'annonces "récentes" utilisées pour l'entraînement
TEST_SIZE = 0.2        # 20% des données réservées à l'évaluation
RANDOM_STATE = 0       # graine fixe pour des résultats reproductibles




def load_dataset(days: int = DAYS_WINDOW) -> pd.DataFrame:
    """Charge et nettoie les annonces des `days` derniers jours."""
    conn = init_db(DB_PATH)
    rows = get_recent_annonces(conn, days=days)
    conn.close()
    return build_clean_dataframe(rows)


def build_full_pipeline() -> Pipeline:
    """
    Étend le Pipeline de preprocessing (imputation + encoding) avec le
    modèle comme dernière étape. Un seul .fit() / .predict() suffit
    ensuite pour tout le flux, de bout en bout.
    """
    preprocessing = build_preprocessing_pipeline()

    return Pipeline([
        *preprocessing.steps,
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )),
    ])


def train_model():
    df = load_dataset()
    print(f"{len(df)} annonces loaded for training.")

    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    X = df.drop(columns=["price", "id"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = build_full_pipeline()
    # pipeline.fit() rejoue automatiquement, dans l'ordre : imputer.fit +
    # transform, target_encoder.fit + transform, onehot.fit + transform,
    # puis model.fit — le tout en apprenant uniquement sur X_train/y_train.
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    Path("models").mkdir(exist_ok=True)
    # Le pipeline entier (preprocessing + modèle) est sauvegardé en un
    # seul objet : imputation, target encoding et one-hot restent
    # attachés au modèle, pas besoin de fichier séparé pour les encoders.
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved in {MODEL_PATH}")
    print(f"Training length: {len(X_train)} | Test length: {len(X_test)}")
    print(f"MAE: {mae:.0f} € | R²: {r2:.3f}")

    return pipeline, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_model()