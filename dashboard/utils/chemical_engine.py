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
    # FIX: Cargar el dataset químico.
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
    # 1. Determinar el feature de entrada inicial que corresponde al target final
    initial_feature_to_exclude = target.replace('target_', '')

    # 2. Excluir columnas siempre (heatid, todos los target_*, y el feature inicial si aplica)
    exclude_cols = ['heatid', target]
    exclude_cols += [col for col in df.columns if col.startswith('target_')]
    if initial_feature_to_exclude in df.columns:
        exclude_cols.append(initial_feature_to_exclude)

    # 3. Filtrar los features definidos en INPUT_FEATURES que estén en el DF y no estén en exclude_cols
    available = [
        col for col in INPUT_FEATURES
        if col in df.columns and col not in exclude_cols
    ]
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
    # FIX: Cargar el dataset químico.
    df = load_chemical_data()

    # VERIFICACIÓN DE EXISTENCIA DE COLUMNA TARGET
    if target not in df.columns:
        error_msg = (
            f"Error: La columna target '{target}' no se encuentra en el dataset cargado. "
            f"Asegúrese de que su pipeline de procesamiento de datos ha creado esta columna."
        )
        logger.error(error_msg)
        raise KeyError(error_msg)

    # Preparar features: Usar la lista limpia de get_chemical_features si feature_list es None
    if feature_list is None:
        feature_cols = get_chemical_features(df, target)
    else:
        # Asegurar que la lista de features seleccionada por el usuario es válida y aplica exclusiones
        initial_feature_to_exclude = target.replace('target_', '')
        feature_cols = [
            f for f in feature_list
            if f in df.columns and f != target and f != initial_feature_to_exclude
        ]

    X = df[feature_cols].copy()
    y = df[target].copy()

    logger.info(f"Muestras iniciales en dataset: {len(X)}")

    # 1. Eliminar filas donde el target es nulo (única eliminación de filas obligatoria)
    rows_before_target_drop = len(X)
    mask_target = y.notnull()
    X = X[mask_target]
    y = y[mask_target]
    rows_after_target_drop = len(X)

    if rows_after_target_drop < rows_before_target_drop:
        logger.warning(f"Se eliminaron {rows_before_target_drop - rows_after_target_drop} filas debido a target nulo.")


    # 2. FIX: Capping del Target (y) para eliminar Outliers extremos (5%/95% cuantil)
    if len(y) > 100 and y.dtype in [np.float64, np.float32]:
        try:
            # Usamos 5% y 95% para máxima robustez contra el error R2 -700
            lower_bound = y.quantile(0.05)
            upper_bound = y.quantile(0.95)

            outlier_mask = (y >= lower_bound) & (y <= upper_bound)

            # Aplicar el filtro a X y a y
            X = X[outlier_mask]
            y = y[outlier_mask]

            logger.warning(f"Se eliminaron {rows_after_target_drop - len(y)} filas por Outliers extremos en {target} (5%/95% Cuantil).")
            rows_after_target_drop = len(y) # Actualizar para el logging final
        except Exception as e:
            logger.error(f"Error durante el capping de outliers: {e}. Continuando sin capping.")


    # 3. FIX: Imputación de NaNs en features: reemplazar nulos por 0
    X = X.fillna(0)

    # 4. Limpieza final de valores infinitos
    X = X.replace([np.inf, -np.inf], 0)
    y = y.replace([np.inf, -np.inf], np.nan).dropna()

    # FIX CRÍTICO: Usar .loc para asegurar que se selecciona filas por índice de fila.
    X = X.loc[y.index]

    if len(X) == 0:
        raise ValueError(f"El dataset para '{target}' quedó vacío después de la limpieza. No se puede entrenar.")


    logger.info(f"Dataset final de entrenamiento: {len(X)} muestras, {len(feature_cols)} features")
    logger.info(f"Target: {target.upper()}")
    logger.info(f"Estadísticas del Target (Y): Min={y.min():.4f}, Max={y.max():.4f}, Media={y.mean():.4f}")


    # Split train/test
    test_size = test_size or DEFAULT_HYPERPARAMS['test_size']
    random_state = random_state or DEFAULT_HYPERPARAMS['random_state']
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
        # Usar el nombre de target limpio (e.g., 'valc') para el nombre del directorio
        model_name = f"chem_{target.replace('target_', '')}_{model_type}_{timestamp}"

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

        # Guardar tambien en formato pkl para compatibilidad con el dashboard (para la evaluación)
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
        # Usar el nombre de target limpio (e.g., 'valc' en lugar de 'target_valc')
        results_file = chemical_results_dir / f"results_{target.replace('target_', '')}.pkl"
        try:
            with open(results_file, 'wb') as f:
                pickle.dump(results_data, f)
            logger.info(f"Guardado: {results_file}")
        except Exception as e:
            logger.error(f"Error al guardar el archivo .pkl para '{target}': {e}")


def get_trained_chemical_models() -> List[str]:
    """
    Escanea el directorio de modelos y devuelve una lista de modelos quimicos entrenados.
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
    """
    models_dir = get_project_root() / "models"
    model_path = models_dir / model_name / "model.joblib"
    return joblib.load(model_path)


def load_chemical_model_metadata(model_name: str) -> Optional[Dict]:
    """
    Carga los metadatos de un modelo quimico si existen.
    """
    models_dir = get_project_root() / "models"
    metadata_path = models_dir / model_name / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None