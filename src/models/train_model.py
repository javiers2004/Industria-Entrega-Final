"""
Script para entrenar el modelo de prediccion de temperatura.

Este script realiza:
1. Carga del dataset procesado
2. Separacion de features y target
3. Train/test split
4. Entrenamiento de XGBoost Regressor
5. Evaluacion (RMSE, R2)
6. Guardado del modelo con BentoML
7. Visualizacion de importancia de variables

Uso:
    python -m src.models.train_model

Requisitos:
    - Dataset procesado en data/processed/dataset_final_acero.csv
    - Ejecutar primero: python -m src.features.build_features
    - En Mac: brew install libomp (para XGBoost)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError as e:
    XGBOOST_AVAILABLE = False
    XGBOOST_ERROR = str(e)

try:
    import bentoml
    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False


def get_project_root() -> Path:
    """Obtiene la raiz del proyecto."""
    return Path(__file__).parent.parent.parent


def load_dataset(filepath: Path) -> pd.DataFrame:
    """
    Carga el dataset procesado.

    Args:
        filepath: Ruta al archivo CSV

    Returns:
        DataFrame con los datos
    """
    df = pd.read_csv(filepath)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas con separador decimal coma a numerico.

    Args:
        df: DataFrame con posibles columnas con comas

    Returns:
        DataFrame con columnas convertidas a numerico
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == 'object':
            # Intentar convertir reemplazando coma por punto
            try:
                df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"  Convertida columna '{col}' a numerico")
            except Exception:
                pass

    return df


def prepare_data(df: pd.DataFrame):
    """
    Prepara los datos para entrenamiento.

    Args:
        df: DataFrame con el dataset completo

    Returns:
        Tuple (X, y) con features y target
    """
    # Columnas a excluir de los features
    exclude_cols = ['heatid', 'target_temperature']

    # Separar X e y
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()
    y = df['target_temperature'].copy()

    # Convertir columnas con comas a numerico
    print("Convirtiendo tipos de datos...")
    X = convert_to_numeric(X)

    # Rellenar NaNs con 0
    X = X.fillna(0)

    print(f"Features: {len(feature_cols)} columnas")
    print(f"Target: target_temperature")

    return X, y


def train_model(X_train, y_train, params: dict = None):
    """
    Entrena un modelo XGBoost.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        params: Parametros opcionales para XGBoost

    Returns:
        Modelo entrenado
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            f"XGBoost no esta disponible: {XGBOOST_ERROR}\n"
            "En Mac ejecuta: brew install libomp"
        )

    # Parametros por defecto
    default_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_jobs': -1,
        'random_state': 42
    }

    if params:
        default_params.update(params)

    print("Entrenando modelo XGBoost...")
    print(f"Parametros: {default_params}")

    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)

    print("Modelo entrenado correctamente")

    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evalua el modelo en el conjunto de test.

    Args:
        model: Modelo entrenado
        X_test: Features de test
        y_test: Target de test

    Returns:
        Diccionario con metricas
    """
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n" + "=" * 40)
    print("RESULTADOS DEL MODELO")
    print("=" * 40)
    print(f"RMSE (Error medio): {rmse:.2f} C")
    print(f"R2 (Explicabilidad): {r2:.4f}")
    print("=" * 40)

    return {
        'rmse': rmse,
        'r2': r2,
        'predictions': preds
    }


def plot_predictions(y_test, predictions, output_path: Path = None):
    """
    Genera grafico de predicciones vs valores reales.

    Args:
        y_test: Valores reales
        predictions: Predicciones del modelo
        output_path: Ruta opcional para guardar el grafico
    """
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, predictions, alpha=0.5, color='blue', s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Temperatura Real (C)", fontsize=12)
    plt.ylabel("Temperatura Predicha (C)", fontsize=12)
    plt.title(f"Prediccion XGBoost (RMSE: {rmse:.1f}C)", fontsize=14)
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Grafico guardado en: {output_path}")

    plt.show()


def plot_feature_importance(model, feature_names, top_n: int = 10, output_path: Path = None):
    """
    Genera grafico de importancia de variables.

    Args:
        model: Modelo entrenado
        feature_names: Nombres de las features
        top_n: Numero de features a mostrar
        output_path: Ruta opcional para guardar el grafico
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)

    print(f"\nTOP {top_n} VARIABLES MAS INFLUYENTES:")
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance, palette='viridis')
    plt.title("Importancia de Variables en la Prediccion de Temperatura", fontsize=14)
    plt.xlabel("Importancia", fontsize=12)
    plt.ylabel("Variable", fontsize=12)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Grafico guardado en: {output_path}")

    plt.show()


def save_model_bentoml(model, model_name: str = "eaf_temperature_model"):
    """
    Guarda el modelo usando BentoML.

    Args:
        model: Modelo entrenado
        model_name: Nombre para el modelo en BentoML
    """
    if not BENTOML_AVAILABLE:
        print("BentoML no esta disponible. El modelo no se guardara en BentoML.")
        return None

    print(f"\nGuardando modelo en BentoML como '{model_name}'...")
    saved_model = bentoml.xgboost.save_model(model_name, model)
    print(f"Modelo guardado: {saved_model}")

    return saved_model


def train_pipeline(
    test_size: float = 0.2,
    random_state: int = 42,
    save_bentoml: bool = True,
    show_plots: bool = True
) -> dict:
    """
    Pipeline completo de entrenamiento.

    Args:
        test_size: Proporcion del dataset para test
        random_state: Semilla para reproducibilidad
        save_bentoml: Si guardar el modelo en BentoML
        show_plots: Si mostrar los graficos

    Returns:
        Diccionario con modelo, metricas y paths
    """
    project_root = get_project_root()
    data_path = project_root / "data" / "processed" / "dataset_final_acero.csv"
    models_dir = project_root / "models"

    # Verificar que existe el dataset
    if not data_path.exists():
        raise FileNotFoundError(
            f"No existe el dataset: {data_path}\n"
            "Ejecuta primero: python -m src.features.build_features"
        )

    # Crear directorio de modelos
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("PIPELINE DE ENTRENAMIENTO")
    print("=" * 50)

    # 1. Cargar datos
    df = load_dataset(data_path)

    # 2. Preparar datos
    X, y = prepare_data(df)

    # 3. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nTrain: {len(X_train)} muestras")
    print(f"Test: {len(X_test)} muestras")

    # 4. Entrenar modelo
    model = train_model(X_train, y_train)

    # 5. Evaluar
    results = evaluate_model(model, X_test, y_test)

    # 6. Visualizaciones
    if show_plots:
        plot_predictions(
            y_test,
            results['predictions'],
            output_path=models_dir / "predicciones.png"
        )
        plot_feature_importance(
            model,
            X.columns.tolist(),
            output_path=models_dir / "importancia_variables.png"
        )

    # 7. Guardar modelo
    if save_bentoml:
        save_model_bentoml(model)

    return {
        'model': model,
        'metrics': {'rmse': results['rmse'], 'r2': results['r2']},
        'feature_names': X.columns.tolist()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrenar modelo de prediccion de temperatura EAF")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporcion para test (default: 0.2)")
    parser.add_argument("--no-bentoml", action="store_true", help="No guardar en BentoML")
    parser.add_argument("--no-plots", action="store_true", help="No mostrar graficos")

    args = parser.parse_args()

    train_pipeline(
        test_size=args.test_size,
        save_bentoml=not args.no_bentoml,
        show_plots=not args.no_plots
    )
