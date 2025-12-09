"""
Funciones para evaluacion de modelos y extraccion de metricas.
Sin dependencias de Streamlit.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import mean_squared_error, r2_score


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
