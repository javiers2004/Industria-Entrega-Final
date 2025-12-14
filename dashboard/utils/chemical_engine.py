"""
Funciones para entrenamiento y evaluacion de modelos de composicion quimica.
Sin dependencias de src/ - completamente autonomo.
"""
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from dashboard.config import (
    INPUT_FEATURES, CHEMICAL_TARGETS, DEFAULT_HYPERPARAMS,
    MODEL_DISPLAY_NAMES, CHEMICAL_SPECS
)
from dashboard.utils.data_engine import get_project_root
from dashboard.utils.model_engine import calculate_metrics, get_feature_importance

# Configurar logging basico
logger = logging.getLogger(__name__)

# Intentar importar XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def load_chemical_data() -> pd.DataFrame:
    """
    Carga el dataset especifico para modelos quimicos.

    Returns:
        DataFrame con los datos quimicos
    """
    from dashboard.utils.data_engine import load_and_clean_data
    return load_and_clean_data("dataset_final_chemical.csv")


def get_chemical_features(df: pd.DataFrame, target: str) -> List[str]:
    """
    Obtiene la lista de features disponibles para entrenamiento quimico,
    excluyendo el target y columnas no relevantes.

    Parameters:
        df: DataFrame con los datos
        target: Target quimico a predecir

    Returns:
        Lista de features disponibles
    """
    # Columnas a excluir siempre
    exclude_cols = ['heatid', target]
    # Tambien excluir todas las columnas target_*
    exclude_cols += [col for col in df.columns if col.startswith('target_')]

    available = [col for col in df.columns if col not in exclude_cols]
    return available


def train_chemical_model(
    target: str,
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
    Entrena un modelo de prediccion de composicion quimica FINAL.

    Parameters:
    -----------
    target : str - Elemento quimico FINAL a predecir ('target_valc', 'target_valmn', etc.)
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
    df = load_chemical_data()

    # Preparar features (excluyendo el target de los features)
    if feature_list is not None:
        feature_cols = [f for f in feature_list if f in df.columns and f != target]
    else:
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

    # Guardar modelo y metadatos
    model_path = None
    if save_model:
        root_dir = get_project_root()
        models_dir = root_dir / "models"
        models_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"chem_{target}_{model_type}_{timestamp}"

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
            "target": target
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadatos guardados en: {metadata_path}")

        # Guardar tambien en formato pkl para compatibilidad con el dashboard
        importance_df = get_feature_importance(model, feature_cols, model_type)
        _save_chemical_results(
            target=target,
            y_test=y_test,
            y_pred=y_pred,
            importance_df=importance_df,
            metrics=metrics,
            models_dir=models_dir
        )

    return model, metrics, feature_cols, X_test, y_test, y_pred, model_path


def _save_chemical_results(
    target: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    importance_df: Optional[pd.DataFrame],
    metrics: Dict[str, float],
    models_dir: Path
):
    """
    Guarda los resultados del modelo quimico en formato pkl para el dashboard.

    Parameters:
        target: Nombre del target quimico
        y_test: Valores reales de test
        y_pred: Predicciones del modelo
        importance_df: DataFrame con importancia de features
        metrics: Diccionario con metricas
        models_dir: Directorio base de modelos
    """
    chemical_results_dir = models_dir / "chemical_results"
    chemical_results_dir.mkdir(parents=True, exist_ok=True)

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
            logger.info(f"Guardado: {results_file}")
        except Exception as e:
            logger.error(f"Error al guardar el archivo .pkl para '{target}': {e}")


def get_trained_chemical_models() -> List[str]:
    """
    Escanea el directorio de modelos y devuelve una lista de modelos quimicos entrenados.

    Returns:
        Lista de nombres de directorios de modelos quimicos
    """
    import os
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)

    # Buscar subdirectorios que empiecen con "chem_" y contengan model.joblib
    model_dirs = [
        d for d in os.listdir(models_dir)
        if os.path.isdir(models_dir / d)
        and d.startswith("chem_")
        and (models_dir / d / "model.joblib").exists()
    ]

    # Ordenar por fecha de modificacion del directorio (mas recientes primero)
    model_dirs.sort(key=lambda d: os.path.getmtime(models_dir / d), reverse=True)

    return model_dirs


def load_chemical_model(model_name: str):
    """
    Carga un modelo quimico desde su directorio.

    Parameters:
        model_name: Nombre del directorio del modelo

    Returns:
        Modelo cargado
    """
    models_dir = get_project_root() / "models"
    model_path = models_dir / model_name / "model.joblib"
    return joblib.load(model_path)


def load_chemical_model_metadata(model_name: str) -> Optional[Dict]:
    """
    Carga los metadatos de un modelo quimico si existen.

    Parameters:
        model_name: Nombre del directorio del modelo

    Returns:
        Diccionario con metadatos o None
    """
    models_dir = get_project_root() / "models"
    metadata_path = models_dir / model_name / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None
