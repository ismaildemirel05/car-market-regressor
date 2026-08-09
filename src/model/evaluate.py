"""
Évalue un modèle entraîné sur le jeu de test et affiche les métriques
de performance ainsi que les features les plus influentes.

Usage :
    python -m src.model.evaluate
"""

import sys
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.model.train import train_model


def evaluate_model(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("--- Evaluation ---")
    print(f"MAE  : {mae:,.0f} €   (mean absolute error)")
    print(f"RMSE : {rmse:,.0f} €   (root mean squared error)")
    print(f"R²   : {r2:.3f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def show_feature_importance(model, feature_names, top_n: int = 15):
    """Affiche les features qui pèsent le plus dans les prédictions du modèle."""
    importances = sorted(
        zip(feature_names, model.feature_importances_),
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
    model, X_train, X_test, y_train, y_test = train_model()
    evaluate_model(model, X_test, y_test)
    show_feature_importance(model, X_test.columns)
