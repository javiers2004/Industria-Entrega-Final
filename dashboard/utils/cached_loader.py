"""
Wrapper con cache de Streamlit para carga de datos.
"""
from typing import Optional

import pandas as pd
import streamlit as st

from src.scripts.data_loader import load_and_clean_data as _load_data, get_data_path


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


__all__ = ['load_and_clean_data', 'get_data_path']
