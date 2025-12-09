"""
Wrapper con cache de Streamlit para carga de datos y resultados de modelos.
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import streamlit as st

# Importaciones asumidas del core del proyecto
from src.config import CHEMICAL_TARGETS
from src.scripts.data_loader import (
    load_and_clean_data as _load_data,
    get_data_path,
    get_project_root
)


@st.cache_data(ttl=3600)  # Cache con TTL de 1 hora
def load_and_clean_data() -> Optional[pd.DataFrame]:
    """
    Carga y limpia el dataset de acero (version original) con cache de Streamlit.

    Returns:
        DataFrame limpio o None si hay error
    """
    try:
        return _load_data()
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo de datos principal. Asegúrese de que exista. Detalle: {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperado cargando datos: {e}")
        return None


@st.cache_data(ttl=3600)
def load_data_for_eda(file_name: str) -> Optional[pd.DataFrame]:
    """
    Carga el dataset final especificado (ej. dataset_final_temp.csv).

    Parameters:
    -----------
    file_name : str - Nombre del archivo a cargar (ej. 'dataset_final_temp.csv')

    Returns:
        DataFrame limpio o None si hay error
    """
    data_path = get_project_root() / "data" / "processed" / file_name
    if not data_path.exists():
        st.error(f"Error: El archivo '{file_name}' para el EDA no fue encontrado en: {data_path}")
        return None

    try:
        # Asumiendo que estos archivos finales ya estan limpios y usan '.' como separador decimal
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Error cargando el dataset {file_name}: {e}")
        return None


@st.cache_data(ttl=3600)
def load_chemical_results() -> Dict[str, Dict[str, Any]]:
    """
    Carga los resultados pre-calculados (metrics, y_test, y_pred, importance_df)
    para todos los targets quimicos.

    NOTA: El usuario debe implementar la lógica de lectura de archivos
    (ej. JSON para métricas, CSV para predicciones/importancia)
    que fueron guardados por el script train_chemical.py.

    Returns:
        Dict[str, Dict[str, Any]]:
            {target: {'y_test': Series, 'y_pred': Series, 'importance_df': DataFrame, 'metrics': Dict}}
    """
    results = {}

    # Directorio donde se asume que train_chemical.py guarda los resultados
    # Ajuste esta ruta si su script de entrenamiento guarda en otro lugar.
    model_results_dir: Path = get_project_root() / "models" / "chemical_results"

    if not model_results_dir.exists():
        st.warning(f"ADVERTENCIA: Directorio de resultados del modelo químico no encontrado: {model_results_dir}")
        st.warning("Usando datos simulados para la visualización.")

        # --- [FALLBACK SIMULADO] ---
        # Simplemente devuelvo datos simulados si no se encuentra el directorio de resultados
        from dashboard.tabs.tab_chemical import load_chemical_results as dummy_loader
        return dummy_loader()
        # --------------------------

    st.info("Cargando resultados de modelos químicos desde archivos...")

    for target in CHEMICAL_TARGETS:
        try:
            # 1. Cargar Metricas (Asumiendo que se guardaron en un JSON)
            metrics_path = model_results_dir / f'{target}_metrics.json'
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            # 2. Cargar Predicciones (Asumiendo que se guardaron y_test/y_pred en un CSV)
            preds_path = model_results_dir / f'{target}_predictions.csv'
            df_preds = pd.read_csv(preds_path)

            # 3. Cargar Importancia de Variables (Asumiendo que se guardó en un CSV)
            importance_path = model_results_dir / f'{target}_importance.csv'
            df_importance = pd.read_csv(importance_path)

            # Almacenar en la estructura requerida por tab_chemical.py
            results[target] = {
                'y_test': df_preds['y_test'],
                'y_pred': df_preds['y_pred'],
                'importance_df': df_importance.sort_values('Importance', ascending=True),
                'metrics': metrics
            }

        except FileNotFoundError:
            st.error(f"Falta el archivo de resultados para el target '{target}'. Asegúrese de entrenar y guardar los resultados.")
        except Exception as e:
            st.error(f"Error inesperado cargando resultados para '{target}': {e}")

    return results


__all__ = [
    'load_and_clean_data',
    'get_data_path',
    'load_data_for_eda',
    'load_chemical_results'
]