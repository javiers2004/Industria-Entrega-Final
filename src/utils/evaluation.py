"""
Funciones para evaluacion de modelos y extraccion de metricas.
Sin dependencias de Streamlit.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred) -> dict:
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
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))

    return {
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae
    }


def get_feature_importance(model, feature_names: list, model_type: str) -> pd.DataFrame:
    """
    Obtiene la importancia de caracteristicas del modelo.

    Parameters:
    -----------
    model : Modelo entrenado
    feature_names : list - Nombres de las features
    model_type : str - Tipo de modelo ('xgboost', 'random_forest', 'linear')

    Returns:
    --------
    DataFrame con columnas 'Feature' e 'Importance', ordenado ascendente por importancia
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
