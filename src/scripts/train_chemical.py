"""
Script CLI para entrenar modelos de prediccion de composicion quimica.

Uso:
    python -m src.models.train_chemical --target valc [--model xgboost|rf|linear]

Ejemplos:
    python -m src.models.train_chemical --target valc --model xgboost
    python -m src.models.train_chemical --target valmn --model rf --n-estimators 200
    python -m src.models.train_chemical --target valsi --model linear

Targets disponibles: valc, valmn, valsi, valp, vals
"""
import argparse
import json
import pickle # <-- NUEVA IMPORTACIÓN
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.logging_config import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from ..config import (
    INPUT_FEATURES, CHEMICAL_TARGETS, DEFAULT_HYPERPARAMS,
    MODEL_DISPLAY_NAMES, CHEMICAL_SPECS
)
from .data_loader import load_and_clean_data, get_project_root
from .evaluation import calculate_metrics, get_feature_importance


def train_chemical_model(
    target: str,
    model_type: str = 'xgboost',
    n_estimators: int = None,
    max_depth: int = None,
    learning_rate: float = None,
    test_size: float = None,
    random_state: int = None
) -> tuple:
    """
    Entrena un modelo de prediccion de composicion quimica.

    Parameters:
    -----------
    target : str - Elemento quimico a predecir ('valc', 'valmn', 'valsi', 'valp', 'vals')
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
    if target not in CHEMICAL_TARGETS:
        raise ValueError(f"Target debe ser uno de: {CHEMICAL_TARGETS}")

    # Usar defaults si no se especifica
    n_estimators = n_estimators or DEFAULT_HYPERPARAMS['n_estimators']
    max_depth = max_depth or DEFAULT_HYPERPARAMS['max_depth']
    learning_rate = learning_rate or DEFAULT_HYPERPARAMS['learning_rate']
    test_size = test_size or DEFAULT_HYPERPARAMS['test_size']
    random_state = random_state or DEFAULT_HYPERPARAMS['random_state']

    # Cargar datos
    logger.info("Cargando datos...")
    df = load_and_clean_data()

    # Preparar features (excluyendo el target de los features)
    feature_cols = [f for f in INPUT_FEATURES if f in df.columns and f != target]
    X = df[feature_cols].copy()
    y = df[target].copy()

    # Eliminar filas con valores nulos
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    logger.info(f"Dataset: {len(X)} muestras, {len(feature_cols)} features")
    logger.info(f"Target: {target.upper()}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Crear modelo
    logger.info(f"Entrenando modelo: {MODEL_DISPLAY_NAMES.get(model_type, model_type)}")

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

    logger.info(f"RMSE: {metrics['RMSE']:.6f}")
    logger.info(f"R2: {metrics['R2']:.4f}")
    logger.info(f"MAE: {metrics['MAE']:.6f}")

    # Mostrar especificacion si existe
    if target in CHEMICAL_SPECS:
        min_spec, max_spec = CHEMICAL_SPECS[target]
        logger.info(f"Especificacion {target.upper()}: {min_spec} - {max_spec}")

    return model, metrics, feature_cols, X_test, y_test, y_pred


def save_plots(model, feature_names, model_type, target, y_test, y_pred, output_dir: Path):
    """Guarda graficos, métricas y datos de prediccion para el dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Crear el subdirectorio específico para resultados químicos
    chemical_results_dir = output_dir / "chemical_results"
    chemical_results_dir.mkdir(parents=True, exist_ok=True)

    # Calcular métricas y feature importance una vez
    metrics = calculate_metrics(y_test, y_pred)
    importance_df = get_feature_importance(model, feature_names, model_type)

    # --- Seccion de Guardado del Archivo Único .pkl para el Dashboard (NUEVO) ---
    if importance_df is not None:
        results_data = {
            'y_test': y_test,
            'y_pred': y_pred,
            'importance_df': importance_df,
            'metrics': metrics
        }
        results_file = chemical_results_dir / f"results_{target}.pkl"
        try:
            with open(results_file, 'wb') as f:
                pickle.dump(results_data, f)
            logger.info(f"Guardado: {results_file} (Archivo único para Dashboard)")
        except Exception as e:
            logger.error(f"Error al guardar el archivo .pkl para '{target}': {e}")
    else:
        logger.warning(f"No se pudo calcular feature importance para guardar el archivo .pkl para '{target}'.")
    # -----------------------------------------------------------------------------


    # Grafico de importancia de variables
    if importance_df is not None: # Ya calculado arriba
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importance_df.tail(15)
        ax.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
        ax.set_xlabel('Importancia')
        ax.set_title(f'Importancia de Variables - {target.upper()}')
        plt.tight_layout()
        plt.savefig(output_dir / f'importancia_{target}.png', dpi=150)
        plt.close()
        logger.info(f"Guardado: {output_dir / f'importancia_{target}.png'}")

        # Guardar Importancia de Variables para el dashboard (CSV - Original)
        importance_path = chemical_results_dir / f'{target}_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Guardado: {importance_path}")

    # Grafico de predicciones vs reales
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Prediccion Perfecta')
    ax.set_xlabel('Valor Real')
    ax.set_ylabel('Valor Predicho')
    ax.set_title(f'Prediccion vs Real - {target.upper()}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'predicciones_{target}.png', dpi=150)
    plt.close()
    logger.info(f"Guardado: {output_dir / f'predicciones_{target}.png'}")

    # Guardar Predicciones (y_test y y_pred) para el dashboard (CSV - Original)
    df_preds = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    preds_path = chemical_results_dir / f'{target}_predictions.csv'
    df_preds.to_csv(preds_path, index=False)
    logger.info(f"Guardado: {preds_path}")

    # Guardar Métricas para el dashboard (JSON - Original)
    metrics_path = chemical_results_dir / f'{target}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Guardado: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar modelo de prediccion de composicion quimica EAF'
    )
    parser.add_argument(
        '--target', '-t',
        required=True,
        choices=CHEMICAL_TARGETS,
        help=f'Elemento quimico a predecir: {", ".join(CHEMICAL_TARGETS)}'
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

    logger.info("=" * 60)
    logger.info(f"ENTRENAMIENTO DE MODELO QUIMICO - {args.target.upper()}")
    logger.info("=" * 60)

    # Entrenar modelo
    model, metrics, feature_names, X_test, y_test, y_pred = train_chemical_model(
        target=args.target,
        model_type=model_type,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )

    # Guardar graficos y datos
    if not args.no_plots:
        output_dir = get_project_root() / args.output_dir
        save_plots(model, feature_names, model_type, args.target, y_test, y_pred, output_dir)

    logger.info("=" * 60)
    logger.info("Entrenamiento completado!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()