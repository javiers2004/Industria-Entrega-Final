"""
Funciones para carga y limpieza de datos.
Sin dependencias de Streamlit.
"""
from pathlib import Path
import os
import pandas as pd

from ..config import CHEMICAL_COLUMNS


def get_project_root() -> Path:
    """
    Obtiene la raiz del proyecto.

    Returns:
        Path a la raiz del proyecto
    """
    return Path(__file__).resolve().parent.parent.parent


def get_data_path() -> Path:
    """
    Busca el archivo de datos usando multiples estrategias.

    Returns:
        Path al archivo dataset_final_acero.csv
    """
    # Estrategia 1: Relativo al proyecto
    PROJECT_ROOT = get_project_root()
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_final_acero.csv"

    if DATA_PATH.exists():
        return DATA_PATH

    # Estrategia 2: Desde working directory
    ALT_DATA_PATH = Path(os.getcwd()) / "data" / "processed" / "dataset_final_acero.csv"
    if ALT_DATA_PATH.exists():
        return ALT_DATA_PATH

    # Estrategia 3: Buscar en padres del working directory
    for parent in Path(os.getcwd()).parents:
        candidate = parent / "data" / "processed" / "dataset_final_acero.csv"
        if candidate.exists():
            return candidate

    return DATA_PATH


def load_and_clean_data() -> pd.DataFrame:
    """
    Carga y limpia el dataset de acero.
    Convierte columnas quimicas de formato europeo (coma) a float.

    Returns:
        DataFrame limpio

    Raises:
        FileNotFoundError: Si no se encuentra el archivo de datos
    """
    data_path = get_data_path()

    if not data_path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset en: {data_path}")

    df = pd.read_csv(data_path)

    # Convertir columnas quimicas de formato europeo (coma) a float
    for col in CHEMICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df
