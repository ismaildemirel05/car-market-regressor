"""
Évalue un modèle entraîné sur le jeu de test et affiche les métriques
de performance ainsi que les features les plus influentes.

Usage :
    python -m src.model.evaluate
"""

import sys
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.model.train import train_model


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> dict:
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("--- Evaluation ---")
    print(f"MAE  : {mae:,.0f} €   (mean absolute error)")
    print(f"RMSE : {rmse:,.0f} €   (root mean squared error)")
    print(f"R²   : {r2:.3f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def show_feature_importance(pipeline: Pipeline, X_test, top_n: int = 15):
    """
    Affiche les features qui pèsent le plus dans les prédictions du modèle.

    `pipeline` contient imputation + encoding + modèle : les vraies
    colonnes vues par le RandomForest (ex : car_brand_BMW,
    car_brand_Peugeot...) n'existent qu'après le passage par les étapes
    de preprocessing, donc on les reconstruit en repassant X_test dans
    tout le pipeline sauf sa dernière étape (le modèle).
    """
    model = pipeline.named_steps["model"]
    preprocessed_X_test = pipeline[:-1].transform(X_test)

    importances = sorted(
        zip(preprocessed_X_test.columns, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print(f"\n--- Top {top_n} features les plus importantes ---")
    for name, importance in importances[:top_n]:
        print(f"{name:<30} {importance:.4f}")


if __name__ == "__main__":
    # On ré-entraîne pour récupérer X_test/y_test au passage : le split n'est
    # pas persisté sur disque, donc évaluer un modèle déjà sauvegardé sans
    # ré-entraîner nécessiterait de sauvegarder aussi le jeu de test à part.
    pipeline, X_train, X_test, y_train, y_test = train_model()
    evaluate_model(pipeline, X_test, y_test)
    show_feature_importance(pipeline, X_test)