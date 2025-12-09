"""
Script CLI para entrenar el modelo de prediccion de temperatura.

Uso:
    python -m src.models.train_temperature [--model xgboost|rf|linear] [--save-bentoml]

Ejemplos:
    python -m src.models.train_temperature --model xgboost --n-estimators 200 --save-bentoml
    python -m src.models.train_temperature --model rf --max-depth 10
    python -m src.models.train_temperature --model linear
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import bentoml
    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False

from ..config import INPUT_FEATURES, DEFAULT_HYPERPARAMS, MODEL_DISPLAY_NAMES
from ..utils.data_loader import load_and_clean_data, get_project_root
from ..utils.evaluation import calculate_metrics, get_feature_importance


def train_temperature_model(
    model_type: str = 'xgboost',
    n_estimators: int = None,
    max_depth: int = None,
    learning_rate: float = None,
    test_size: float = None,
    random_state: int = None
) -> tuple:
    """
    Entrena un modelo de prediccion de temperatura.

    Parameters:
    -----------
    model_type : str - Tipo de modelo ('xgboost', 'random_forest', 'linear')
    n_estimators : int - Numero de estimadores (para tree models)
    max_depth : int - Profundidad maxima (para tree models)
    learning_rate : float - Learning rate (para XGBoost)
    test_size : float - Proporcion de datos para test
    random_state : int - Semilla para reproducibilidad

    Returns:
    --------
    tuple: (model, metrics, feature_names, X_test, y_test, y_pred)
    """
    # Usar defaults si no se especifica
    n_estimators = n_estimators or DEFAULT_HYPERPARAMS['n_estimators']
    max_depth = max_depth or DEFAULT_HYPERPARAMS['max_depth']
    learning_rate = learning_rate or DEFAULT_HYPERPARAMS['learning_rate']
    test_size = test_size or DEFAULT_HYPERPARAMS['test_size']
    random_state = random_state or DEFAULT_HYPERPARAMS['random_state']

    # Cargar datos
    print("Cargando datos...")
    df = load_and_clean_data()

    # Preparar features y target
    feature_cols = [f for f in INPUT_FEATURES if f in df.columns]
    X = df[feature_cols].copy()
    y = df['target_temperature'].copy()

    # Eliminar filas con valores nulos
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    print(f"Dataset: {len(X)} muestras, {len(feature_cols)} features")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Crear modelo
    print(f"Entrenando modelo: {MODEL_DISPLAY_NAMES.get(model_type, model_type)}")

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost no esta instalado. Instalar con: pip install xgboost")
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Modelo no reconocido: {model_type}")

    # Entrenar
    model.fit(X_train, y_train)

    # Predecir y evaluar
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2: {metrics['R2']:.4f}")
    print(f"MAE: {metrics['MAE']:.2f}")

    return model, metrics, feature_cols, X_test, y_test, y_pred


def save_plots(model, feature_names, model_type, y_test, y_pred, output_dir: Path):
    """Guarda graficos de importancia y predicciones."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Grafico de importancia de variables
    importance_df = get_feature_importance(model, feature_names, model_type)
    if importance_df is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importance_df.tail(15)
        ax.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
        ax.set_xlabel('Importancia')
        ax.set_title('Importancia de Variables - Temperatura')
        plt.tight_layout()
        plt.savefig(output_dir / 'importancia_temperatura.png', dpi=150)
        plt.close()
        print(f"Guardado: {output_dir / 'importancia_temperatura.png'}")

    # Grafico de predicciones vs reales
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Prediccion Perfecta')
    ax.set_xlabel('Valor Real')
    ax.set_ylabel('Valor Predicho')
    ax.set_title('Prediccion vs Real - Temperatura')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'predicciones_temperatura.png', dpi=150)
    plt.close()
    print(f"Guardado: {output_dir / 'predicciones_temperatura.png'}")


def save_to_bentoml(model, model_type: str):
    """Guarda el modelo en BentoML."""
    if not BENTOML_AVAILABLE:
        raise ImportError("BentoML no esta instalado. Instalar con: pip install bentoml")

    model_name = "eaf_temperature_model"
    saved_model = bentoml.sklearn.save_model(
        model_name,
        model,
        signatures={"predict": {"batchable": True}},
        metadata={
            "model_type": model_type,
            "target": "temperature"
        }
    )
    print(f"Modelo guardado en BentoML: {saved_model.tag}")
    return saved_model


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar modelo de prediccion de temperatura EAF'
    )
    parser.add_argument(
        '--model', '-m',
        choices=['xgboost', 'rf', 'linear'],
        default='xgboost',
        help='Tipo de modelo (default: xgboost)'
    )
    parser.add_argument(
        '--n-estimators', '-n',
        type=int,
        default=DEFAULT_HYPERPARAMS['n_estimators'],
        help=f'Numero de estimadores (default: {DEFAULT_HYPERPARAMS["n_estimators"]})'
    )
    parser.add_argument(
        '--max-depth', '-d',
        type=int,
        default=DEFAULT_HYPERPARAMS['max_depth'],
        help=f'Profundidad maxima (default: {DEFAULT_HYPERPARAMS["max_depth"]})'
    )
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=DEFAULT_HYPERPARAMS['learning_rate'],
        help=f'Learning rate para XGBoost (default: {DEFAULT_HYPERPARAMS["learning_rate"]})'
    )
    parser.add_argument(
        '--save-bentoml', '-b',
        action='store_true',
        help='Guardar modelo en BentoML'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='models',
        help='Directorio para guardar graficos (default: models/)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='No generar graficos'
    )

    args = parser.parse_args()

    # Mapear 'rf' a 'random_forest'
    model_type = 'random_forest' if args.model == 'rf' else args.model

    print("=" * 60)
    print("ENTRENAMIENTO DE MODELO DE TEMPERATURA EAF")
    print("=" * 60)

    # Entrenar modelo
    model, metrics, feature_names, X_test, y_test, y_pred = train_temperature_model(
        model_type=model_type,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )

    # Guardar graficos
    if not args.no_plots:
        output_dir = get_project_root() / args.output_dir
        save_plots(model, feature_names, model_type, y_test, y_pred, output_dir)

    # Guardar en BentoML
    if args.save_bentoml:
        save_to_bentoml(model, model_type)

    print("=" * 60)
    print("Entrenamiento completado!")
    print("=" * 60)


if __name__ == '__main__':
    main()
