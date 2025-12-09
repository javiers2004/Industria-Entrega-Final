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
# Importamos el loader dummy (simulado)
from dashboard.tabs.tab_chemical import load_chemical_results as dummy_loader


@st.cache_data(ttl=3600)  # Mantenemos el cache para los datos originales
def load_and_clean_data() -> Optional[pd.DataFrame]:
    """
    Carga y limpia el dataset de acero con cache de Streamlit (datos originales).
    """
    try:
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


# --- IMPORTANTE: Se elimina el decorador @st.cache_data para forzar la ejecución ---
def load_chemical_results() -> Dict[str, Dict[str, Any]]:
    """
    Carga los resultados pre-calculados (metrics, y_test, y_pred, importance_df)
    para todos los targets quimicos.
    """
    results = {}

    # Directorio donde train_chemical.py guardó los resultados, usando la ruta relativa del proyecto
    try:
        model_results_dir: Path = get_project_root() / "models" / "chemical_results"
    except Exception as e:
        # Esto captura errores si la importación o ejecución de get_project_root falla
        st.error(f"ERROR FATAL DE RUTA: No se pudo determinar la raíz del proyecto. {e}")
        return dummy_loader()

    # --- LÍNEAS DE DEBUG/VERIFICACIÓN ---
    st.info(f"DEBUG: Buscando resultados en: {model_results_dir.resolve()}")

    if not model_results_dir.exists():
        st.error("DEBUG: Directorio de resultados NO ENCONTRADO.")
        st.warning(f"ADVERTENCIA: Directorio no encontrado. Usando datos simulados.")
        # Llama a la funcion que devuelve los datos simulados
        return dummy_loader()

    st.success("DEBUG: Directorio ENCONTRADO. Cargando datos reales.")
    st.info("Cargando resultados de modelos químicos desde archivos...")
    # --- FIN LÍNEAS DE DEBUG/VERIFICACIÓN ---

    # Se ejecuta este bloque SOLO si existe el directorio de resultados.
    for target in CHEMICAL_TARGETS:
        try:
            # 1. Cargar Metricas (JSON)
            metrics_path = model_results_dir / f'{target}_metrics.json'
            # Verificación extra: si falta un archivo, no cargues ese target.
            if not metrics_path.exists():
                 st.warning(f"ADVERTENCIA: Archivo de métricas faltante para {target}.")
                 continue

            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            # 2. Cargar Predicciones (CSV)
            preds_path = model_results_dir / f'{target}_predictions.csv'
            df_preds = pd.read_csv(preds_path)

            # 3. Cargar Importancia de Variables (CSV)
            importance_path = model_results_dir / f'{target}_importance.csv'
            df_importance = pd.read_csv(importance_path)

            results[target] = {
                'y_test': df_preds['y_test'],
                'y_pred': df_preds['y_pred'],
                'importance_df': df_importance.sort_values('Importance', ascending=True),
                'metrics': metrics
            }

        except FileNotFoundError:
            st.error(f"Falta un archivo de resultados para el target '{target}'. Verifique {model_results_dir}.")
        except Exception as e:
            st.error(f"Error inesperado cargando resultados para '{target}': {e}")

    return results


__all__ = [
    'load_and_clean_data',
    'get_data_path',
    'load_data_for_eda',
    'load_chemical_results'
]