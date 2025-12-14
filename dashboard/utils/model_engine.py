"""
Funciones para entrenamiento y evaluacion de modelos del dashboard.
Sin dependencias de src/ - completamente autonomo.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from dashboard.config import INPUT_FEATURES, DEFAULT_HYPERPARAMS, MODEL_DISPLAY_NAMES
from dashboard.utils.data_engine import load_and_clean_data, get_project_root

# Configurar logging basico para el dashboard
logger = logging.getLogger(__name__)

# Intentar importar XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series, List[float]],
    y_pred: Union[np.ndarray, pd.Series, List[float]]
) -> Dict[str, float]:
    """
    Calcula metricas de evaluacion para regresion.

    Parameters:
    -----------
    y_true : array-like - Valores reales
    y_pred : array-like - Valores predichos

    Returns:
    --------
    dict con RMSE, R2, MAE
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))

    return {
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae
    }


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    model_type: str
) -> Optional[pd.DataFrame]:
    """
    Obtiene la importancia de caracteristicas del modelo.

    Parameters:
    -----------
    model : Modelo entrenado (LinearRegression, RandomForest, XGBoost)
    feature_names : list - Nombres de las features
    model_type : str - Tipo de modelo ('xgboost', 'random_forest', 'linear')

    Returns:
    --------
    DataFrame con columnas 'Feature' e 'Importance', ordenado ascendente por importancia.
    Retorna None si el tipo de modelo no es reconocido.
    """
    if model_type == 'linear':
        importance = np.abs(model.coef_)
    elif model_type in ['random_forest', 'xgboost']:
        importance = model.feature_importances_
    else:
        return None

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)

    return importance_df


def train_temperature_model(
    model_type: str = 'xgboost',
    n_estimators: int = None,
    max_depth: int = None,
    learning_rate: float = None,
    test_size: float = None,
    random_state: int = None,
    save_model: bool = True,
    feature_list: List[str] = None
) -> Tuple[Any, Dict[str, float], List[str], pd.DataFrame, pd.Series, np.ndarray, Optional[Path]]:
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
    save_model : bool - Si es True, guarda el modelo y los metadatos en disco.
    feature_list : List[str] - Lista de features a usar. Si es None, usa INPUT_FEATURES.

    Returns:
    --------
    tuple: (model, metrics, feature_names, X_test, y_test, y_pred, model_path)
           model_path es None si save_model es False.
    """
    # Usar defaults si no se especifica
    n_estimators = n_estimators or DEFAULT_HYPERPARAMS['n_estimators']
    max_depth = max_depth or DEFAULT_HYPERPARAMS['max_depth']
    learning_rate = learning_rate or DEFAULT_HYPERPARAMS['learning_rate']
    test_size = test_size or DEFAULT_HYPERPARAMS['test_size']
    random_state = random_state or DEFAULT_HYPERPARAMS['random_state']

    # Cargar datos
    logger.info("Cargando datos...")
    df = load_and_clean_data()

    # Preparar features y target
    # Usar feature_list si se proporciona, sino INPUT_FEATURES
    if feature_list is not None:
        feature_cols = [f for f in feature_list if f in df.columns]
    else:
        feature_cols = [f for f in INPUT_FEATURES if f in df.columns]
    X = df[feature_cols].copy()
    y = df['target_temperature'].copy()

    # Eliminar filas con valores nulos
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    logger.info(f"Dataset: {len(X)} muestras, {len(feature_cols)} features")

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

    logger.info(f"RMSE: {metrics['RMSE']:.2f}")
    logger.info(f"R2: {metrics['R2']:.4f}")
    logger.info(f"MAE: {metrics['MAE']:.2f}")

    # Guardar modelo y metadatos
    model_path = None
    if save_model:
        root_dir = get_project_root()
        models_dir = root_dir / "models"
        models_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"temp_{model_type}_{timestamp}"

        # Crear subdirectorio para este modelo
        model_subdir = models_dir / model_name
        model_subdir.mkdir(exist_ok=True)

        model_path = model_subdir / "model.joblib"
        metadata_path = model_subdir / "metadata.json"

        # Guardar modelo
        joblib.dump(model, model_path)
        logger.info(f"Modelo guardado en: {model_path}")

        # Guardar metadatos (features, hiperparametros, metricas)
        metadata = {
            "model_type": model_type,
            "features": feature_cols,
            "hyperparameters": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "test_size": test_size,
                "random_state": random_state
            },
            "metrics": metrics,
            "timestamp": timestamp,
            "target": "target_temperature"
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadatos guardados en: {metadata_path}")

    return model, metrics, feature_cols, X_test, y_test, y_pred, model_path
