"""
Wrapper con cache de Streamlit para carga de datos.
"""
import os
from pathlib import Path
from typing import Optional

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


__all__ = ['load_and_clean_data', 'load_data_for_eda', 'get_data_path']
