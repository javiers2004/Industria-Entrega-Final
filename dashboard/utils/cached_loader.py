"""
Wrapper con cache de Streamlit para carga de datos.
"""
import os
import pickle # <-- NUEVA IMPORTACIÓN
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import streamlit as st

from src.scripts.data_loader import load_and_clean_data as _load_data, get_data_path, get_project_root


@st.cache_data(ttl=3600)  # Cache con TTL de 1 hora
def load_and_clean_data() -> Optional[pd.DataFrame]:
    """
    Carga y limpia el dataset de acero con cache de Streamlit.

    Returns:
        DataFrame limpio o None si hay error
    """
    try:
        return _load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return None
    except Exception as e:
        st.error(f"Error inesperado cargando datos: {e}")
        return None


@st.cache_data(ttl=3600)  # Cache con TTL de 1 hora
def load_data_for_eda(file_name: str) -> Optional[pd.DataFrame]:
    """
    Carga un dataset especifico para EDA con cache de Streamlit.

    Parameters:
        file_name: Nombre del archivo CSV (ej: 'dataset_final_temp.csv')

    Returns:
        DataFrame o None si hay error
    """
    try:
        # Estrategia 1: Usar la raiz del proyecto
        project_root = get_project_root()
        file_path = project_root / "data" / "processed" / file_name

        if file_path.exists():
            df = pd.read_csv(file_path)
            return df

        # Estrategia 2: Desde working directory
        alt_path = Path(os.getcwd()) / "data" / "processed" / file_name
        if alt_path.exists():
            df = pd.read_csv(alt_path)
            return df

        # Estrategia 3: Buscar en padres del working directory
        for parent in Path(os.getcwd()).parents:
            candidate = parent / "data" / "processed" / file_name
            if candidate.exists():
                df = pd.read_csv(candidate)
                return df

        st.error(f"Archivo no encontrado: {file_name}")
        return None

    except FileNotFoundError as e:
        st.error(f"Archivo no encontrado: {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperado cargando datos: {e}")
        return None


@st.cache_data(ttl=3600)
def load_single_chemical_result(target_name: str) -> Optional[Dict[str, Any]]:
    """
    Carga los resultados pre-calculados del modelo de composición química
    para un target específico (ej: 'C', 'Mn').

    Se asume que el archivo está en la carpeta 'models/chemical_results/'
    y se llama 'results_{target_name}.pkl'.

    Parameters:
        target_name: El componente químico cuyo resultado se desea cargar.

    Returns:
        Diccionario de resultados (y_test, y_pred, importance_df, metrics)
        o None si hay error.
    """
    file_name = f"results_{target_name}.pkl"
    # El script de entrenamiento guarda en models/chemical_results/
    sub_dir = Path("models") / "chemical_results"

    try:
        search_paths = [get_project_root() / sub_dir, Path(os.getcwd()) / sub_dir] + \
                       [parent / sub_dir for parent in Path(os.getcwd()).parents]

        for base_path in search_paths:
            file_path = base_path / file_name
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    # El archivo .pkl debe contener un diccionario con las keys esperadas
                    return pickle.load(f)

        st.error(f"Archivo de resultados de modelo no encontrado para '{target_name}'. Se buscó: {file_name} en las rutas esperadas (ej: models/chemical_results/).")
        return None

    except Exception as e:
        st.error(f"Error inesperado cargando los resultados del modelo para '{target_name}': {e}")
        return None


__all__ = ['load_and_clean_data', 'load_data_for_eda', 'load_single_chemical_result', 'get_data_path']