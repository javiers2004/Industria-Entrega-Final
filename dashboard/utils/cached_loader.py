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
# Importamos el loader dummy (simulado). Esta importación asume que
# dashboard/tabs/tab_chemical.py ya contiene la función load_chemical_results
# que simula los datos.
from dashboard.tabs.tab_chemical import load_chemical_results as dummy_loader


@st.cache_data(ttl=3600)  # Cache con TTL de 1 hora
def load_and_clean_data() -> Optional[pd.DataFrame]:
    """
    Carga y limpia el dataset de acero con cache de Streamlit (datos originales).

    Returns:
        DataFrame limpio o None si hay error
    """
    try:
        # Llama a la funcion original de carga y limpieza de datos
        return _load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return None
    except Exception as e:
        st.error(f"Error inesperado cargando datos: {e}")
        return None


@st.cache_data(ttl=3600)
def load_data_for_eda(file_name: str) -> Optional[pd.DataFrame]:
    """
    Carga el dataset final especificado (ej. dataset_final_temp.csv) para el EDA.

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
    """
    results = {}

    # Directorio donde train_chemical.py guardó los resultados
    model_results_dir: Path = get_project_root() / "models" / "chemical_results"

    # --- LÍNEAS DE DEBUG CRÍTICAS ---
    # Esto mostrará la ruta exacta que Streamlit está buscando.
    st.info(f"DEBUG: Intentando cargar resultados desde: {model_results_dir.resolve()}")

    if not model_results_dir.exists():
        st.error("DEBUG: Directorio NO ENCONTRADO. Recurriendo a datos simulados.")
        st.warning(f"ADVERTENCIA: Directorio de resultados del modelo químico no encontrado.")
        st.warning("Usando datos simulados para la visualización.")
        return dummy_loader()

    st.success("DEBUG: El directorio de resultados FUE ENCONTRADO. Intentando cargar archivos reales.")
    st.info("Cargando resultados de modelos químicos desde archivos...")

    # Se ejecuta este bloque SOLO si existe el directorio de resultados.
    for target in CHEMICAL_TARGETS:
        try:
            # 1. Cargar Metricas (JSON)
            metrics_path = model_results_dir / f'{target}_metrics.json'
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            # 2. Cargar Predicciones (CSV)
            preds_path = model_results_dir / f'{target}_predictions.csv'
            df_preds = pd.read_csv(preds_path)

            # 3. Cargar Importancia de Variables (CSV)
            importance_path = model_results_dir / f'{target}_importance.csv'
            df_importance = pd.read_csv(importance_path)

            # Almacenar en la estructura requerida
            results[target] = {
                'y_test': df_preds['y_test'],
                'y_pred': df_preds['y_pred'],
                'importance_df': df_importance.sort_values('Importance', ascending=True),
                'metrics': metrics
            }

        except FileNotFoundError:
            st.error(f"Falta el archivo de resultados para el target '{target}'. Asegúrese de entrenar y guardar los resultados.")
            # Si un target falla, devolvemos un conjunto vacío para ese target y continuamos.
        except Exception as e:
            st.error(f"Error inesperado cargando resultados para '{target}': {e}")

    return results


__all__ = [
    'load_and_clean_data',
    'get_data_path',
    'load_data_for_eda',
    'load_chemical_results'
]